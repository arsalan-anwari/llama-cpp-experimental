#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Shared helpers for the toy bitwise-nn demo architecture.
inline const std::vector<std::string> & bitwise_demo_vocab_words() {
    static const std::vector<std::string> words = {
        "okay ", "hmm ", "bitwise ", "logic ", "ping ", "pong ", "left ", "right ",
        "hash ", "shift ", "xnor ", "demo ", "question? ", "answer: ", "maybe ",
        "why ", "because ", "cool ", "noise ", "++ ", "-- ", ":: ", ">> ", "<< ",
        "end. ", "uh ", "huh ", "huh? ", "sure ", "\n"
    };
    return words;
}

inline uint32_t bitwise_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline char bitwise_to_printable(uint32_t x) {
    return static_cast<char>(32 + (x % 95)); // ASCII 32..126
}

inline uint64_t bitwise_fnv1a64(const std::vector<uint8_t> & data) {
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t b : data) {
        hash ^= b;
        hash *= 1099511628211ull;
    }
    return hash;
}
