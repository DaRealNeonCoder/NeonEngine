 #version 450

//comments are annoying.
layout(location = 0) in vec3 fragColor;          
layout(location = 1) in vec3 fragPosWorld;      
layout(location = 2) in vec3 fragNormalWorld;   
layout (location = 3) in vec3 inViewVec;        
layout (location = 4) in vec3 inLightVec;       
layout(location = 5) in vec4 inShadowCoord;     
layout(location = 6) in vec2 uv;     
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 0, binding = 2) uniform sampler2D tex1; // prolly should have a separate pipeline for this.

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0, std140) uniform GlobalUbo {
    mat4 projection;
    mat4 inverseProjection;
    mat4 view;
    mat4 inverseView;

    vec4 ambientLightColor;

    PointLight pointLights[1];
    vec4 numLights;

    mat4 lightSpaceMatrix;
    vec4 lightDir;
} ubo;
layout(push_constant) uniform Push {
mat4 modelMatrix;
mat4 normalMatrix;
vec4 thisIsStupid;
} push;

layout (constant_id = 0) const int enablePCF = 1;

layout (constant_id = 1) const float specShininess = 32.0;

const float AMBIENT_FALLBACK = 0.1;

float textureProj(vec4 shadowCoord, vec2 off, float bias)
{
	float shadow = 1.0;

    if (shadowCoord.z > -0.0 && shadowCoord.z < 1.0)
	{
		float dist = texture( shadowMap, shadowCoord.st + off , 1).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z) 
		{
			shadow = 0;
		}
	}
	return shadow;
}


float filterPCF(vec4 sc, float bias)
{
	ivec2 texDim = textureSize(shadowMap, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y), bias);
			count++;
		}
	
	}
	return shadowFactor / count;
}

void main() {
    vec3 ambient = ubo.ambientLightColor.xyz * ubo.ambientLightColor.w;
    vec3 diffuseLight = vec3(0.0);
    vec3 specularLight = vec3(0.0);
    vec3 surfaceNormal = normalize(fragNormalWorld);

    vec3 cameraPosWorld = ubo.inverseView[3].xyz;
    vec3 viewDirection = normalize(cameraPosWorld - fragPosWorld);

    
    vec4 sc = inShadowCoord / inShadowCoord.w;

    float shadow = 0.0;
    float bias = max(0.005 * (1.0 - dot(fragNormalWorld, ubo.lightDir.xyz)), 0.001);
    if (1 == 1) {
        shadow = filterPCF(sc, bias * 5);
    } else {
        shadow = textureProj(sc, vec2(0.0), bias);
    }
    PointLight light = ubo.pointLights[0];
    vec3 directionToLight = light.position.xyz - fragPosWorld;
    float attenuation = 1.0 / dot(directionToLight, directionToLight); // distance squared
    directionToLight = normalize(directionToLight);

    float cosAngIncidence = max(dot(surfaceNormal, directionToLight), 0);
    vec3 intensity = light.color.xyz * light.color.w * attenuation;

    diffuseLight += intensity * cosAngIncidence;

    // specular lighting
    vec3 halfAngle = normalize(directionToLight + viewDirection);
    float blinnTerm = dot(surfaceNormal, halfAngle);
    blinnTerm = clamp(blinnTerm, 0, 1);
    blinnTerm = pow(blinnTerm, 512.0); // higher values -> sharper highlight
    specularLight += intensity * blinnTerm;
  
    outColor = vec4(ambient + (diffuseLight * fragColor)  * shadow, 1.0);
    //outColor = vec4((ambient + (diffuseLight * fragColor + specularLight * fragColor) * shadow), 1.0);
    
    if(push.thisIsStupid.x ==67.f)
    {
    
    vec2 newUv=(uv * 4) + vec2(0.5,0);

    vec3 color = texture(tex1, newUv).xyz;
	outColor = vec4(color,1.0f);
    }
}


