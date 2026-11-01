import slangml_common;

// Diffusion training losses [SI 3.7.1, Alg 28].
//
//   sml_loss_weighted_rigid_align_fwd_f32   xAligned = weighted Kabsch(xGt -> xPred)
//   sml_loss_diffusion_mse_fwd_f32          perStructure[s] = (1/(3A)) sum_l w_l |xPred - xAligned|^2
//   sml_loss_diffusion_mse_bwd_f32          gradX = gradScale[s] * (2/(3A)) * w * (xPred - xAligned)
//   sml_loss_bond_fwd_f32                   perBond[b] = | (x_l - x_m) - (g_l - g_m) |^2
//   sml_loss_bond_bwd_f32                   per-bond gradient contributions (host scatter-adds)
//
// The host forms the full diffusion loss:
//   L_diffusion = (t^2 + sigma^2) / (t + sigma)^2 * (L_MSE + lambda_bond * L_bond) + L_smoothLDDT
// (SI Eq. 6), and the aligned target is stop-gradient (Alg 28 line 12).
//
// Weighted Kabsch (Alg 28).  With weights w_l, centroids mu_p, mu_g:
//   M = sum_l w_l (p_l - mu_p)(g_l - mu_g)^T  = U S V^T   (3x3 SVD)
//   R = U V^T;  if det(R) < 0 flip the sign of V's column for the smallest s_k
//   aligned_l = R (g_l - mu_g) + mu_p
// Eigenvectors of M^T M come from cyclic Jacobi (12 sweeps, fp32).  This runs
// in thread 0 of each workgroup; it is 3x3 and not on the critical path of the
// rest of the step.
//
// Layout (fp32): positions [A, 3] per structure; weights [A]; structures are
// stacked along the outer index.  One workgroup (256 threads) per structure.
// Atom count per structure is a hot scalar; the batch is folded into dispatch.

struct KabschParams
{
    KernelHeader header;
    uint structures;
    uint atomsPerStructure;
    uint reserved0;
    uint reserved1;
};

struct MSEParams
{
    KernelHeader header;
    uint structures;
    uint atomsPerStructure;
    uint reserved0;
    uint reserved1;
};

struct BondParams
{
    KernelHeader header;
    uint bonds;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlDiffusionLossPush;

groupshared float smlLossScratch[256];
groupshared float smlKabschM[9];
groupshared float smlKabschR[9];

float sml_loss_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_loss_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

float sml_loss_reduce(float value, uint lane)
{
    float result = slangml_workgroup_reduce_add_f32_256(value, smlLossScratch, lane);
    GroupMemoryBarrierWithGroupSync();
    return result;
}

// Cyclic Jacobi for a symmetric 3x3 matrix a (row-major, destroyed).
// Returns eigenvectors as columns of v (row-major) and eigenvalues in ev.
void sml_loss_jacobi3(inout float a[9], out float v[9], out float ev[3])
{
    for (uint i = 0u; i < 9u; ++i)
        v[i] = (i % 4u == 0u) ? 1.0f : 0.0f;

    for (uint sweep = 0u; sweep < 12u; ++sweep)
    {
        float off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if (off < 1.0e-30f)
            break;

        for (uint pair = 0u; pair < 3u; ++pair)
        {
            uint pp = pair == 2u ? 1u : 0u;
            uint qq = pair == 0u ? 1u : 2u;
            float apq = a[pp * 3u + qq];
            if (abs(apq) < 1.0e-20f)
                continue;

            float theta = (a[qq * 3u + qq] - a[pp * 3u + pp]) / (2.0f * apq);
            float sgn = theta >= 0.0f ? 1.0f : -1.0f;
            float t = sgn / (abs(theta) + sqrt(theta * theta + 1.0f));
            float c = rsqrt(t * t + 1.0f);
            float s = t * c;

            // A <- A J  (columns p and q)
            for (uint k = 0u; k < 3u; ++k)
            {
                float akp = a[k * 3u + pp];
                float akq = a[k * 3u + qq];
                a[k * 3u + pp] = c * akp - s * akq;
                a[k * 3u + qq] = s * akp + c * akq;
            }
            // A <- J^T A  (rows p and q)
            for (uint k = 0u; k < 3u; ++k)
            {
                float apk = a[pp * 3u + k];
                float aqk = a[qq * 3u + k];
                a[pp * 3u + k] = c * apk - s * aqk;
                a[qq * 3u + k] = s * apk + c * aqk;
            }
            // V <- V J
            for (uint k = 0u; k < 3u; ++k)
            {
                float vkp = v[k * 3u + pp];
                float vkq = v[k * 3u + qq];
                v[k * 3u + pp] = c * vkp - s * vkq;
                v[k * 3u + qq] = s * vkp + c * vkq;
            }
        }
    }
    ev[0] = a[0];
    ev[1] = a[4];
    ev[2] = a[8];
}

// Returns R = U V^T for the 3x3 matrix M (row-major), with det(R) = +1.
float3x3 sml_loss_kabsch_rotation(float m[9])
{
    // A = M^T M
    float a[9];
    for (uint r = 0u; r < 3u; ++r)
        for (uint c = 0u; c < 3u; ++c)
        {
            float acc = 0.0f;
            for (uint k = 0u; k < 3u; ++k)
                acc = fma(m[k * 3u + r], m[k * 3u + c], acc);
            a[r * 3u + c] = acc;
        }

    float v[9];
    float ev[3];
    sml_loss_jacobi3(a, v, ev);

    // Sort eigenpairs in descending order (3 elements).
    for (uint pass = 0u; pass < 2u; ++pass)
        for (uint k = 0u; k < 2u; ++k)
            if (ev[k] < ev[k + 1u])
            {
                float tmpEv = ev[k];
                ev[k] = ev[k + 1u];
                ev[k + 1u] = tmpEv;
                for (uint r = 0u; r < 3u; ++r)
                {
                    float tmp = v[r * 3u + k];
                    v[r * 3u + k] = v[r * 3u + k + 1u];
                    v[r * 3u + k + 1u] = tmp;
                }
            }

    // Singular values and left singular vectors u_k = M v_k / s_k.
    float s[3];
    float u[9];
    for (uint k = 0u; k < 3u; ++k)
        s[k] = sqrt(max(ev[k], 0.0f));

    float eps = 1.0e-7f;
    for (uint k = 0u; k < 2u; ++k)
    {
        float3 col = float3(0.0f);
        if (s[k] > eps)
        {
            for (uint r = 0u; r < 3u; ++r)
            {
                float acc = 0.0f;
                for (uint c = 0u; c < 3u; ++c)
                    acc = fma(m[r * 3u + c], v[c * 3u + k], acc);
                col[r] = acc / s[k];
            }
        }
        else
        {
            col = k == 0u ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        }
        for (uint r = 0u; r < 3u; ++r)
            u[r * 3u + k] = col[r];
    }
    // Make u1 orthogonal to u0, then u2 completes a right-handed basis when
    // the third singular value is degenerate.
    float3 u0 = float3(u[0], u[3], u[6]);
    float3 u1 = float3(u[1], u[4], u[7]);
    u1 = normalize(u1 - dot(u1, u0) * u0);
    float3 u2 = s[2] > eps
        ? float3(
            m[0] * v[2] + m[1] * v[5] + m[2] * v[8],
            m[3] * v[2] + m[4] * v[5] + m[5] * v[8],
            m[6] * v[2] + m[7] * v[5] + m[8] * v[8]) / s[2]
        : cross(u0, u1);
    u2 = normalize(u2);
    u[0] = u0.x; u[3] = u0.y; u[6] = u0.z;
    u[1] = u1.x; u[4] = u1.y; u[7] = u1.z;
    u[2] = u2.x; u[5] = u2.y; u[8] = u2.z;

    // R = U V^T.  If det(R) < 0, flip the sign of the smallest-singular-value
    // column of V and recompute (Alg 28 lines 7-9).
    float3x3 rot;
    for (uint attempt = 0u; attempt < 2u; ++attempt)
    {
        for (uint r = 0u; r < 3u; ++r)
            for (uint c = 0u; c < 3u; ++c)
            {
                float acc = 0.0f;
                for (uint k = 0u; k < 3u; ++k)
                    acc = fma(u[r * 3u + k], v[c * 3u + k], acc);
                rot[r][c] = acc;
            }
        float det = determinant(rot);
        if (det >= 0.0f)
            break;
        for (uint r = 0u; r < 3u; ++r)
            v[r * 3u + 2u] = -v[r * 3u + 2u];
    }
    return rot;
}

// One workgroup per structure.  Computes the weighted Kabsch alignment of xGt
// onto xPred and writes xAligned.  Inputs: xPred[S*A*3], xGt[S*A*3], w[S*A].
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_loss_weighted_rigid_align_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    KabschParams* p = reinterpret<KabschParams*>(smlDiffusionLossPush.headerAddress);
    uint structure = p.header.dispatchBase.x + gid.x;
    if (structure >= p.structures)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc xGt = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc xAligned = slangml_tensor_desc(p.header, 3u);

    uint A = p.atomsPerStructure;
    uint atomBase = structure * A;
    uint lane = tid.x;

    // Pass 1: total weight and weighted centroids.
    float sumW = 0.0f;
    float3 sumP = float3(0.0f);
    float3 sumG = float3(0.0f);
    for (uint a = lane; a < A; a += 256u)
    {
        float wa = sml_loss_load(w, atomBase + a);
        sumW += wa;
        float3 pa = float3(sml_loss_load(xPred, (atomBase + a) * 3u + 0u),
                           sml_loss_load(xPred, (atomBase + a) * 3u + 1u),
                           sml_loss_load(xPred, (atomBase + a) * 3u + 2u));
        float3 ga = float3(sml_loss_load(xGt, (atomBase + a) * 3u + 0u),
                           sml_loss_load(xGt, (atomBase + a) * 3u + 1u),
                           sml_loss_load(xGt, (atomBase + a) * 3u + 2u));
        sumP += wa * pa;
        sumG += wa * ga;
    }
    float totalW = sml_loss_reduce(sumW, lane);
    float3 muP;
    float3 muG;
    for (uint c = 0u; c < 3u; ++c)
    {
        muP[c] = sml_loss_reduce(sumP[c], lane) / totalW;
        muG[c] = sml_loss_reduce(sumG[c], lane) / totalW;
    }

    // Pass 2: cross-covariance M = sum_l w_l (p - mu_p)(g - mu_g)^T.
    float cov[9];
    for (uint i = 0u; i < 9u; ++i)
        cov[i] = 0.0f;
    for (uint a = lane; a < A; a += 256u)
    {
        float wa = sml_loss_load(w, atomBase + a);
        float3 pa = float3(sml_loss_load(xPred, (atomBase + a) * 3u + 0u),
                           sml_loss_load(xPred, (atomBase + a) * 3u + 1u),
                           sml_loss_load(xPred, (atomBase + a) * 3u + 2u)) - muP;
        float3 ga = float3(sml_loss_load(xGt, (atomBase + a) * 3u + 0u),
                           sml_loss_load(xGt, (atomBase + a) * 3u + 1u),
                           sml_loss_load(xGt, (atomBase + a) * 3u + 2u)) - muG;
        for (uint r = 0u; r < 3u; ++r)
            for (uint c = 0u; c < 3u; ++c)
                cov[r * 3u + c] = fma(wa * pa[r], ga[c], cov[r * 3u + c]);
    }
    for (uint i = 0u; i < 9u; ++i)
    {
        float total = sml_loss_reduce(cov[i], lane);
        if (lane == 0u)
            smlKabschM[i] = total;
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u)
    {
        float m[9];
        for (uint i = 0u; i < 9u; ++i)
            m[i] = smlKabschM[i];
        float3x3 rot = sml_loss_kabsch_rotation(m);
        for (uint r = 0u; r < 3u; ++r)
            for (uint c = 0u; c < 3u; ++c)
                smlKabschR[r * 3u + c] = rot[r][c];
    }
    GroupMemoryBarrierWithGroupSync();

    // Pass 3: aligned_l = R (g_l - mu_g) + mu_p.
    float rot00 = smlKabschR[0]; float rot01 = smlKabschR[1]; float rot02 = smlKabschR[2];
    float rot10 = smlKabschR[3]; float rot11 = smlKabschR[4]; float rot12 = smlKabschR[5];
    float rot20 = smlKabschR[6]; float rot21 = smlKabschR[7]; float rot22 = smlKabschR[8];
    for (uint a = lane; a < A; a += 256u)
    {
        uint base = (atomBase + a) * 3u;
        float3 gc = float3(sml_loss_load(xGt, base + 0u),
                           sml_loss_load(xGt, base + 1u),
                           sml_loss_load(xGt, base + 2u)) - muG;
        float ax = rot00 * gc.x + rot01 * gc.y + rot02 * gc.z + muP.x;
        float ay = rot10 * gc.x + rot11 * gc.y + rot12 * gc.z + muP.y;
        float az = rot20 * gc.x + rot21 * gc.y + rot22 * gc.z + muP.z;
        sml_loss_store(xAligned, base + 0u, ax);
        sml_loss_store(xAligned, base + 1u, ay);
        sml_loss_store(xAligned, base + 2u, az);
    }
}

// perStructure[s] = (1/(3A)) * sum_l w_l |xPred_l - aligned_l|^2.  One workgroup per structure.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_loss_diffusion_mse_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    MSEParams* p = reinterpret<MSEParams*>(smlDiffusionLossPush.headerAddress);
    uint structure = p.header.dispatchBase.x + gid.x;
    if (structure >= p.structures)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc aligned = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc perStructure = slangml_tensor_desc(p.header, 3u);

    uint A = p.atomsPerStructure;
    uint atomBase = structure * A;
    float acc = 0.0f;
    for (uint a = tid.x; a < A; a += 256u)
    {
        uint base = (atomBase + a) * 3u;
        float3 d = float3(sml_loss_load(xPred, base + 0u) - sml_loss_load(aligned, base + 0u),
                          sml_loss_load(xPred, base + 1u) - sml_loss_load(aligned, base + 1u),
                          sml_loss_load(xPred, base + 2u) - sml_loss_load(aligned, base + 2u));
        acc = fma(sml_loss_load(w, atomBase + a), dot(d, d), acc);
    }
    float total = sml_loss_reduce(acc, tid.x);
    if (tid.x == 0u)
        sml_loss_store(perStructure, structure, total / (3.0f * float(A)));
}

// gradX_l = gradScale[s] * (2/(3A)) * w_l * (xPred_l - aligned_l).  The aligned target is stop-gradient.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_loss_diffusion_mse_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MSEParams* p = reinterpret<MSEParams*>(smlDiffusionLossPush.headerAddress);
    uint perAtom = p.atomsPerStructure;
    uint total = p.structures * perAtom * 3u;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc aligned = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc gradScale = slangml_tensor_desc(p.header, 3u);
    TensorDesc gradX = slangml_tensor_desc(p.header, 4u);

    uint coord = (p.header.dispatchBase.x + dtid.x) % 3u;
    uint atom = (p.header.dispatchBase.x + dtid.x) / 3u;
    uint structure = atom / perAtom;
    float scale = sml_loss_load(gradScale, structure) * 2.0f / (3.0f * float(perAtom));
    float diff = sml_loss_load(xPred, (p.header.dispatchBase.x + dtid.x)) - sml_loss_load(aligned, (p.header.dispatchBase.x + dtid.x));
    sml_loss_store(gradX, (p.header.dispatchBase.x + dtid.x), scale * sml_loss_load(w, atom) * diff);
}

// perBond[b] = | (x_l - x_m) - (g_l - g_m) |^2.  bondPairs: [B, 2] uint.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_loss_bond_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    BondParams* p = reinterpret<BondParams*>(smlDiffusionLossPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.bonds)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc xGt = slangml_tensor_desc(p.header, 1u);
    TensorDesc bondPairs = slangml_tensor_desc(p.header, 2u);
    TensorDesc perBond = slangml_tensor_desc(p.header, 3u);

    uint l = slangml_bda_load_u32(slangml_final_byte_address(bondPairs, int((p.header.dispatchBase.x + dtid.x) * 2u), 4u));
    uint m = slangml_bda_load_u32(slangml_final_byte_address(bondPairs, int((p.header.dispatchBase.x + dtid.x) * 2u + 1u), 4u));
    float3 r = float3(sml_loss_load(xPred, l * 3u + 0u) - sml_loss_load(xPred, m * 3u + 0u),
                      sml_loss_load(xPred, l * 3u + 1u) - sml_loss_load(xPred, m * 3u + 1u),
                      sml_loss_load(xPred, l * 3u + 2u) - sml_loss_load(xPred, m * 3u + 2u))
        - float3(sml_loss_load(xGt, l * 3u + 0u) - sml_loss_load(xGt, m * 3u + 0u),
                 sml_loss_load(xGt, l * 3u + 1u) - sml_loss_load(xGt, m * 3u + 1u),
                 sml_loss_load(xGt, l * 3u + 2u) - sml_loss_load(xGt, m * 3u + 2u));
    sml_loss_store(perBond, (p.header.dispatchBase.x + dtid.x), dot(r, r));
}

// Per-bond gradient contributions.  contrib[b, 0, :] = +2 * scale_b * r_b,
// contrib[b, 1, :] = -2 * scale_b * r_b.  The host scatter-adds into gradX.
// Bonds can share atoms, so the scatter is done on the host (no float atomics).
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_loss_bond_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    BondParams* p = reinterpret<BondParams*>(smlDiffusionLossPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.bonds)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc xGt = slangml_tensor_desc(p.header, 1u);
    TensorDesc bondPairs = slangml_tensor_desc(p.header, 2u);
    TensorDesc scale = slangml_tensor_desc(p.header, 3u);
    TensorDesc contrib = slangml_tensor_desc(p.header, 4u);

    uint l = slangml_bda_load_u32(slangml_final_byte_address(bondPairs, int((p.header.dispatchBase.x + dtid.x) * 2u), 4u));
    uint m = slangml_bda_load_u32(slangml_final_byte_address(bondPairs, int((p.header.dispatchBase.x + dtid.x) * 2u + 1u), 4u));
    float s2 = 2.0f * sml_loss_load(scale, (p.header.dispatchBase.x + dtid.x));
    for (uint c = 0u; c < 3u; ++c)
    {
        float r = (sml_loss_load(xPred, l * 3u + c) - sml_loss_load(xPred, m * 3u + c))
            - (sml_loss_load(xGt, l * 3u + c) - sml_loss_load(xGt, m * 3u + c));
        sml_loss_store(contrib, (p.header.dispatchBase.x + dtid.x) * 6u + c, s2 * r);
        sml_loss_store(contrib, (p.header.dispatchBase.x + dtid.x) * 6u + 3u + c, -s2 * r);
    }
}
