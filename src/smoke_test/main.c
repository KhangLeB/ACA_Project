/* Sanity check that M-extension multiply works under rv64imc; returns 0 on success. */
int main(void) {
    volatile int a = 6, b = 7;
    int c = a * b;
    return (c == 42) ? 0 : 1;
}
