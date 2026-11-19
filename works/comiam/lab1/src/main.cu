#include <cstdio>

int main() {
    std::fputs("Synthetic runtime failure\n", stderr);
    return 1;
}
