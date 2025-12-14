# llama-cpp-experimental

## Custom bitwise-nn demo 

This branch adds a toy “bitwise-nn” architecture with a fully custom inference path to show how you can bypass the normal llama graph and still reuse GGUF storage and the CLI. The "LLM" makes absolute no sense and is psudo random, its just to show how you can add you own inference path for your custom model architecture.

### What the inference path does
- Takes the raw input tokens as bytes (`I_tk`).
- Splits them in half; left half is always smaller or equal.
- Hashes each half with FNV-1a 64 → maps into a u16 range `[min(I_tk), max(I_tk)]` to get `H_l` and `H_r`.
- Loads tensors `A` and `B`, applies zero-fill right-shifts from `demo.shiftA`/`demo.shiftB`, adds `H_l`/`H_r`, then XNORs them into `C`.
- Samples a tiny built-in vocab using values from `C` to produce a playful response.

### How to run the demo
```
./build-x64-linux-gcc-release/bin/llama-cli \
  -m models/custom/bitwise-nn/bitwise-example-1-U16.gguf \
  -c 256
```

### Example conversation
```
./bin/llama-cli -m models/custom/bitwise-nn/bitwise-example-1-U16.gguf -c 256 -t 4  ✔ 
bitwise-nn demo ready.
type your message: hello what do you do actualy?

> Vright {N<< cool << uquh question? left right .demo Ghmm logic @answer: okay hmm right 
0cool answer: because maybe sure why right demo uh }answer: (huh question? tmdemo <pong >> -- huh? left 

type your message: Okay thats weird... are you doing okay?

> right >> hash :: demo question? -- rn:: huh why &
uh 
xnor << /<< BBZright rxnor hmm huh? ouh demo 
uh << logic :: noise pshift answer: noise cool "because logic shift why shift 

type your message: okay, i am leaving you be. good luck?

> ++ xnor hmm #++ pong huh okay uh z++ o)logic demo :: cool uh :: bitwise qmaybe end. cool okay demo logic ++ xnor right qwhy hash demo why 7ping pong >> logic cool cool 5answer: 6question? << 5

type your message: ^C
```
