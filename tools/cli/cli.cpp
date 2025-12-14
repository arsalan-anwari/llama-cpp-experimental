#include "common.h"
#include "arg.h"
#include "console.h"
// #include "log.h"

#include "server-context.h"
#include "server-task.h"

#include "gguf.h"
#include "ggml.h"
#include "../src/bitwise-nn.h"

#include <atomic>
#include <fstream>
#include <thread>
#include <signal.h>
#include <execinfo.h>
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

static void segv_handler(int) {
    void * bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n--- SIGSEGV (llama-cli) ---\n");
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    _exit(128 + SIGSEGV);
}

#define BITWISE_QK_U16 16
struct bitwise_block_u16 {
    ggml_fp16_t d;
    ggml_fp16_t m;
    uint16_t qs[BITWISE_QK_U16];
};

static std::vector<uint16_t> bitwise_tensor_codes(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<uint16_t> out(n, 0);

    switch (t->type) {
        case GGML_TYPE_QU16_0:
            {
                const bitwise_block_u16 * blocks = reinterpret_cast<const bitwise_block_u16 *>(t->data);
                int64_t idx = 0;
                const int nb = (int) ((n + BITWISE_QK_U16 - 1)/BITWISE_QK_U16);
                for (int b = 0; b < nb && idx < n; ++b) {
                    for (int i = 0; i < BITWISE_QK_U16 && idx < n; ++i) {
                        out[idx++] = blocks[b].qs[i];
                    }
                }
            } break;
        case GGML_TYPE_F16:
            {
                const ggml_fp16_t * vals = reinterpret_cast<const ggml_fp16_t *>(t->data);
                for (int64_t i = 0; i < n; ++i) {
                    out[i] = (uint16_t) std::llround(ggml_fp16_to_fp32(vals[i]));
                }
            } break;
        case GGML_TYPE_F32:
            {
                const float * vals = reinterpret_cast<const float *>(t->data);
                for (int64_t i = 0; i < n; ++i) {
                    out[i] = (uint16_t) std::llround(vals[i]);
                }
            } break;
        default:
            {
                const uint16_t * vals = reinterpret_cast<const uint16_t *>(t->data);
                const size_t count = (size_t) std::min<int64_t>(n, ggml_nbytes(t) / sizeof(uint16_t));
                std::copy(vals, vals + count, out.begin());
            } break;
    }

    return out;
}

static std::vector<llama_token> bitwise_generate_plan(
        const std::vector<uint16_t> & C,
        int64_t cols,
        uint16_t H_l,
        uint16_t H_r,
        uint32_t byte_vocab_size) {
    const auto & vocab_words = bitwise_demo_vocab_words();
    const llama_token word_base = (llama_token) byte_vocab_size;

    const size_t rows = cols > 0 ? C.size() / (size_t) cols : 0;
    const int64_t safe_cols = cols > 0 ? cols : 1;
    const size_t out_chars = std::min<size_t>(48, std::max<size_t>(8, C.size()));

    uint32_t state = bitwise_mix32((uint32_t(H_l) << 16) ^ uint32_t(H_r) ^ 0x9e3779b9u);

    std::vector<llama_token> plan;
    plan.reserve(out_chars + 4);

    for (size_t t = 0; t < out_chars; ) {
        const uint8_t r = rows ? (uint8_t)((t + (state & 15u)) % rows) : 0;
        const uint8_t c = (uint8_t)(((t >> 1) + ((state >> 8) & 15u)) % safe_cols);

        const uint16_t logit = C.empty() ? 0 : C[r * safe_cols + c];
        state = bitwise_mix32(state ^ (uint32_t(logit) * 2654435761u) ^ (uint32_t) t);

        if ((state & 3u) != 0) {
            plan.push_back(word_base + (state % vocab_words.size()));
        } else {
            const char ch = bitwise_to_printable(state ^ logit);
            plan.push_back((llama_token) (uint8_t) ch);
        }

        ++t;
        if ((state & 511u) == 0 && plan.size() > 40) {
            break;
        }
    }

    if (plan.empty()) {
        plan.push_back(word_base);
    }

    return plan;
}

static bool maybe_run_bitwise_cli(const common_params & params) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    gguf_context * ctx = gguf_init_from_file(params.model.path.c_str(), ip);
    if (!ctx) {
        return false;
    }

    const int64_t arch_k = gguf_find_key(ctx, "general.architecture");
    const char * arch_name = arch_k >= 0 ? gguf_get_val_str(ctx, arch_k) : "";
    if (!arch_name || std::string(arch_name) != "bitwise-nn") {
        gguf_free(ctx);
        ggml_free(ctx_data);
        return false;
    }

    const int32_t shiftA = gguf_find_key(ctx, "demo.shiftA") >= 0 ? gguf_get_val_i32(ctx, gguf_find_key(ctx, "demo.shiftA")) : 0;
    const int32_t shiftB = gguf_find_key(ctx, "demo.shiftB") >= 0 ? gguf_get_val_i32(ctx, gguf_find_key(ctx, "demo.shiftB")) : 0;

    ggml_tensor * A = ggml_get_tensor(ctx_data, "A");
    ggml_tensor * B = ggml_get_tensor(ctx_data, "B");
    ggml_tensor * Cref = ggml_get_tensor(ctx_data, "C");
    if (!A || !B || !Cref) {
        fprintf(stderr, "bitwise-nn: missing tensors\n");
        gguf_free(ctx);
        ggml_free(ctx_data);
        return true;
    }

    auto A_vals = bitwise_tensor_codes(A);
    auto B_vals = bitwise_tensor_codes(B);

    const size_t n_vals = std::min(A_vals.size(), B_vals.size());
    A_vals.resize(n_vals);
    B_vals.resize(n_vals);

    const auto shift_apply = [](std::vector<uint16_t> & v, int sh) {
        if (sh <= 0) return;
        if (sh >= 16) { std::fill(v.begin(), v.end(), 0); return; }
        for (auto & x : v) x = (uint16_t)(x >> sh);
    };
    shift_apply(A_vals, shiftA);
    shift_apply(B_vals, shiftB);

    std::vector<uint16_t> C_vals(n_vals, 0);

    const uint32_t byte_vocab = 256;

    fprintf(stdout, "bitwise-nn demo ready.\n");
    fprintf(stdout, "type your message: ");
    fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<uint8_t> I_tk(line.begin(), line.end());
        const size_t left_len  = I_tk.size() / 2;
        const size_t right_len = I_tk.size() - left_len;

        std::vector<uint8_t> left(I_tk.begin(), I_tk.begin() + left_len);
        std::vector<uint8_t> right(I_tk.begin() + left_len, I_tk.begin() + left_len + right_len);

        uint8_t vmin = 0, vmax = 0;
        if (!I_tk.empty()) {
            vmin = vmax = I_tk[0];
            for (uint8_t v : I_tk) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
        } else {
            vmax = 255;
        }

        auto map_hash = [&](uint64_t h) {
            if (vmax == vmin) return (uint16_t) vmin;
            const uint16_t span = (uint16_t)(vmax - vmin);
            return (uint16_t)(vmin + (h % (uint64_t)(span + 1)));
        };

        const uint16_t H_l = map_hash(bitwise_fnv1a64(left));
        const uint16_t H_r = map_hash(bitwise_fnv1a64(right));

        for (size_t i = 0; i < n_vals; ++i) {
            const uint16_t a = (uint16_t)(A_vals[i] + H_l);
            const uint16_t b = (uint16_t)(B_vals[i] + H_r);
            C_vals[i] = (uint16_t)(~(a ^ b));
        }

        const int64_t cols = Cref->ne[0] ? Cref->ne[0] : 1;
        auto plan = bitwise_generate_plan(C_vals, cols, H_l, H_r, byte_vocab);

        std::string out;
        const auto & vocab_words = bitwise_demo_vocab_words();
        for (llama_token tok : plan) {
            if (tok < (llama_token) byte_vocab) {
                out.push_back((char) tok);
            } else {
                size_t idx = (size_t)(tok - byte_vocab) % vocab_words.size();
                out += vocab_words[idx];
            }
        }

        fprintf(stdout, "\n> %s\n\n", out.c_str());
        fprintf(stdout, "type your message: ");
        fflush(stdout);
    }

    gguf_free(ctx);
    ggml_free(ctx_data);
    return true;
}
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress
        defaults.oaicompat_chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    }

    std::string generate_completion(result_timings & out_timings) {
        server_response_reader rd = ctx_server.get_response_reader();
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id        = rd.get_new_id();
            task.index     = 0;
            task.params    = defaults;    // copy
            task.cli_input = messages;    // copy
            task.cli_files = input_files; // copy
            rd.post_task({std::move(task)});
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }
};

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    common_init();

    // short-circuit for custom bitwise-nn demo
    if (maybe_run_bitwise_cli(params)) {
        return 0;
    }

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
    struct sigaction sigsegv_action;
    sigsegv_action.sa_handler = segv_handler;
    sigemptyset (&sigsegv_action.sa_mask);
    sigsegv_action.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sigsegv_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    ctx_cli.ctx_server.init();

    console::spinner::stop();
    console::log("\n");

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_info();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    if (!params.system_prompt.empty()) {
        ctx_cli.messages.push_back({
            {"role",    "system"},
            {"content", params.system_prompt}
        });
    }

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /read               add a text file\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("\n");

    // interactive loop
    std::string cur_msg;
    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            std::string marker = ctx_cli.load_input_file(fname, false);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded text from '%s'\n", fname.c_str());
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second, timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
