# llama-cpp-experimental::example_custom_quantization_scheme

This branch gives a detailed example how you implement your own quantization scheme to convert f32 weights into your own type. For this example `f32 --> u16` is used, but it can be adapted to do anything realy. 

You can quantize like this:
```bash
./bin/llama-quantize models/custom/bitwise-nn/bitwise-example-1.gguf models/custom/bitwise-nn/bitwise-example-1-U16.gguf QU16_0
```

```bash
main: build = 7375 (e39a2ce6)
main: built with GNU 15.2.1 for Linux x86_64
main: quantizing 'models/custom/bitwise-nn/bitwise-example-1.gguf' to 'models/custom/bitwise-nn/bitwise-example-1-U16.gguf' as QU16_0
llama_model_loader: loaded meta data with 4 key-value pairs and 3 tensors from models/custom/bitwise-nn/bitwise-example-1.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitwise-nn
llama_model_loader: - kv   1:                          general.file_type u32              = 0
llama_model_loader: - kv   2:                                demo.shiftA i32              = 2
llama_model_loader: - kv   3:                                demo.shiftB i32              = 3
llama_model_loader: - type  f32:    3 tensors
[   1/   3]                                    A - [   16,    16,     1,     1], type =    f32, converting to qu16_0 .. size =     0.00 MiB ->     0.00 MiB
[   2/   3]                                    B - [   16,    16,     1,     1], type =    f32, converting to qu16_0 .. size =     0.00 MiB ->     0.00 MiB
[   3/   3]                                    C - [   16,    16,     1,     1], type =    f32, converting to qu16_0 .. size =     0.00 MiB ->     0.00 MiB
llama_model_quantize_impl: model size  =     0.00 MiB
llama_model_quantize_impl: quant size  =     0.00 MiB

main: quantize time =     1.04 ms
main:    total time =     1.04 ms
```

And check the GGUF like this:
```bash
./bin/llama-gguf models/custom/bitwise-nn/bitwise-example-1-U16.gguf r n
```

```bash
gguf_ex_read_0: version:      3
gguf_ex_read_0: alignment:   32
gguf_ex_read_0: data offset: 352
gguf_ex_read_0: n_kv: 5
gguf_ex_read_0: kv[0]: key = general.architecture
gguf_ex_read_0: kv[1]: key = demo.shiftA
gguf_ex_read_0: kv[2]: key = demo.shiftB
gguf_ex_read_0: kv[3]: key = general.quantization_version
gguf_ex_read_0: kv[4]: key = general.file_type
gguf_ex_read_0: find key: some.parameter.string not found.
gguf_ex_read_0: n_tensors: 3
gguf_ex_read_0: tensor[0]: name = A, size = 576, offset = 0
gguf_ex_read_0: tensor[1]: name = B, size = 576, offset = 576
gguf_ex_read_0: tensor[2]: name = C, size = 576, offset = 1152
gguf_ex_read_1: version:      3
gguf_ex_read_1: alignment:   32
gguf_ex_read_1: data offset: 352
gguf_ex_read_1: n_kv: 5
gguf_ex_read_1: kv[0]: key = general.architecture
gguf_ex_read_1: kv[1]: key = demo.shiftA
gguf_ex_read_1: kv[2]: key = demo.shiftB
gguf_ex_read_1: kv[3]: key = general.quantization_version
gguf_ex_read_1: kv[4]: key = general.file_type
gguf_ex_read_1: n_tensors: 3
gguf_ex_read_1: tensor[0]: name = A, size = 576, offset = 0, type = qu16_0, n_elts = 16
gguf_ex_read_1: tensor[1]: name = B, size = 576, offset = 576, type = qu16_0, n_elts = 16
gguf_ex_read_1: tensor[2]: name = C, size = 576, offset = 1152, type = qu16_0, n_elts = 16
gguf_ex_read_1: reading tensor 0 data
gguf_ex_read_1: tensor[0]: n_dims = 2, ne = (16, 16, 1, 1), name = A, data = 0x55b7c8cb6930
A data[:10] : 47743 18089 62087 19448 40622 12197 11329 48049 19299 2588 

gguf_ex_read_1: reading tensor 1 data
gguf_ex_read_1: tensor[1]: n_dims = 2, ne = (16, 16, 1, 1), name = B, data = 0x55b7c8cb6b70
B data[:10] : 65246 43306 31988 45153 0 17578 37411 42836 59139 24344 

gguf_ex_read_1: reading tensor 2 data
gguf_ex_read_1: tensor[2]: n_dims = 2, ne = (16, 16, 1, 1), name = C, data = 0x55b7c8cb6db0
C data[:10] : 0 0 0 0 0 0 0 0 0 0 

gguf_ex_read_1: ctx_data size: 3200
```