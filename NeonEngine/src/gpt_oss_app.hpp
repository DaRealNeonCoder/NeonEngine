#pragma once

namespace lve {

// Command-line runner for the gpt-oss model (headless, no window):
//
//   LveEngine.exe --gpt-oss <checkpoint/original dir> [options]
//
//   --tokens 1,2,3        prompt token ids (comma or space separated)
//   --tokens-file f.txt   prompt token ids from a file
//   --max-new N           tokens to generate greedily (default 32)
//   --stop 200002,...     stop token ids (default: <|return|> 200002, <|call|> 200012)
//   --ctx N               maximum context length (default 4096)
//   --dense bf16|q8       attention / LM-head weight format (default bf16)
//   --experts auto|device|host   where the MXFP4 experts live (default auto)
//   --score               no generation: run every prompt token with the LM head on
//   --dump-logits f.bin   write fp32 logits [positions][vocab] for each scored position
//   --shaders DIR         shaders/ML directory (default ./shaders/ML)
//   --device NAME         pick a GPU by name substring
//
// Memorilla (off unless --memorilla is given; shaders/ML/docs/memorilla_plan.md):
//   --memorilla DIR       memory module (memorilla.json + memorilla.safetensors)
//   --mem-store F.nmem    document store; the module's K memory tokens go before the prompt
//   --mem-k N             use only the first N memory tokens
//   --mem-query ids / --mem-query-file f   query tokens for z_q (default: the prompt)
//   --mem-query-vec f     z_q as raw fp32 (stores made by an external encoder)
//   --mem-dump f          write z_q and the memory tokens (fp32), for testing
//   --soft-prefix f       raw fp32 input vectors [n][hidden] fed before the prompt (testing)
//
// Token ids are the o200k_harmony tokenizer's; tokenisation is not built in yet
// (see shaders/ML/tests/gpt_oss_tokens.py).
int runGptOssCli(int argc, char** argv);

}  // namespace lve
