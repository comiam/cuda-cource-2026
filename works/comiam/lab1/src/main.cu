#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "submission --input FILE --output FILE [--mode coalesced|noncoalesced]\n"
                     "Input: N alpha, then N pairs x y. Output: N triples add, SAXPY, ReLU.\n";
        return 0;
    }

    const char *input = nullptr;
    const char *output = nullptr;
    bool valid = argc % 2 == 1;
    for (int i = 1; valid && i < argc; i += 2) {
        const std::string flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--input" && value.substr(0, 2) != "--") {
            input = argv[i + 1];
        } else if (flag == "--output" && value.substr(0, 2) != "--") {
            output = argv[i + 1];
        } else if (flag != "--mode" || (value != "coalesced" && value != "noncoalesced")) {
            valid = false;
        }
    }
    if (!valid || input == nullptr || output == nullptr) {
        std::cerr << "Invalid arguments\n";
        return 2;
    }

    std::ifstream in(input);
    int count = 0;
    float alpha = 0;
    if (!(in >> count >> alpha) || count < 0) {
        std::cerr << "Invalid input header\n";
        return 2;
    }
    for (int i = 0; i < count; ++i) {
        float x = 0;
        float y = 0;
        if (!(in >> x >> y)) {
            std::cerr << "Invalid input pair\n";
            return 2;
        }
    }

    std::ofstream out(output);
    for (int i = 0; i < count && out; ++i) {
        // Deliberately wrong: ReLU cannot produce a negative result.
        out << "0 0 -1\n";
    }
    out.close();
    if (!out) {
        std::cerr << "Cannot write output\n";
        return 1;
    }
    return 0;
}
