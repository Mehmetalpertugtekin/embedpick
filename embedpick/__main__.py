from .cli import main

# Guard şart: bu satır modül düzeyinde çalışırsa, dosyayı import eden
# herhangi bir şey (test toplama, dokümantasyon aracı) süreci öldürür.
if __name__ == "__main__":
    raise SystemExit(main())
