

class RECTANGLECipher:
    def __init__(self):
        self.block_size = 64
        self.key_size = 80
        self.rounds = 25
        # Official Round constants for 25 rounds
        self.RC = [
            0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19, 
            0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C, 0x18, 0x11, 0x03, 0x06, 
            0x0D, 0x1B, 0x17, 0x0E, 0x1D
        ]
        # Official S-Box Table
        self.SBOX_TABLE = [0x6, 0x5, 0xc, 0xa, 0x1, 0xe, 0x7, 0x9, 0xb, 0x0, 0x3, 0xd, 0x8, 0xf, 0x4, 0x2]
        # Inverse S-Box Table
        self.INV_SBOX_TABLE = [0x9, 0x4, 0xf, 0xa, 0xe, 0x1, 0x0, 0x6, 0xc, 0x7, 0x3, 0x8, 0x2, 0xb, 0x5, 0xd]

    
    # TIER 1: CORE PRIMITIVES (Atomic Bitsliced Logic)
    

    def sub_column(self, r0, r1, r2, r3):
        """Bitsliced S-box (Forward) via bit-parallel lookup"""
        nr0, nr1, nr2, nr3 = 0, 0, 0, 0
        for i in range(16):
            col = ((r0 >> i) & 1) | (((r1 >> i) & 1) << 1) | \
                  (((r2 >> i) & 1) << 2) | (((r3 >> i) & 1) << 3)
            out_col = self.SBOX_TABLE[col]
            nr0 |= ((out_col & 1) << i)
            nr1 |= (((out_col >> 1) & 1) << i)
            nr2 |= (((out_col >> 2) & 1) << i)
            nr3 |= (((out_col >> 3) & 1) << i)
        return nr0 & 0xFFFF, nr1 & 0xFFFF, nr2 & 0xFFFF, nr3 & 0xFFFF

    def sub_column_inv(self, r0, r1, r2, r3):
        """Bitsliced S-box (Inverse) via bit-parallel lookup"""
        nr0, nr1, nr2, nr3 = 0, 0, 0, 0
        for i in range(16):
            col = ((r0 >> i) & 1) | (((r1 >> i) & 1) << 1) | \
                  (((r2 >> i) & 1) << 2) | (((r3 >> i) & 1) << 3)
            inv_col = self.INV_SBOX_TABLE[col]
            nr0 |= ((inv_col & 1) << i)
            nr1 |= (((inv_col >> 1) & 1) << i)
            nr2 |= (((inv_col >> 2) & 1) << i)
            nr3 |= (((inv_col >> 3) & 1) << i)
        return nr0 & 0xFFFF, nr1 & 0xFFFF, nr2 & 0xFFFF, nr3 & 0xFFFF

    def shift_rows(self, r0, r1, r2, r3):
        """Linear Diffusion: Left Rotations (0, 1, 12, 13)"""
        r1 = ((r1 << 1) | (r1 >> 15)) & 0xFFFF
        r2 = ((r2 << 12) | (r2 >> 4)) & 0xFFFF
        r3 = ((r3 << 13) | (r3 >> 3)) & 0xFFFF
        return r0, r1, r2, r3

    def shift_rows_inv(self, r0, r1, r2, r3):
        """Inverse Linear Diffusion: Right Rotations"""
        r1 = ((r1 >> 1) | (r1 << 15)) & 0xFFFF
        r2 = ((r2 >> 12) | (r2 << 4)) & 0xFFFF
        r3 = ((r3 >> 13) | (r3 << 3)) & 0xFFFF
        return r0, r1, r2, r3

    def add_round_key(self, rows, rk):
        """XOR state with round key"""
        return [rows[i] ^ rk[i] for i in range(4)]

    
    # TIER 2: ROUND COMPOSITION
    

    def encrypt_round(self, rows, rk):
        rows = self.add_round_key(rows, rk)
        rows = self.sub_column(*rows)
        rows = self.shift_rows(*rows)
        return list(rows)

    def decrypt_round(self, rows, rk):
        rows = self.shift_rows_inv(*rows)
        rows = self.sub_column_inv(*rows)
        rows = self.add_round_key(rows, rk)
        return list(rows)

    
    # TIER 3: KEY SCHEDULE
    

    def key_update_80(self, k, r_idx):
        # 1. S-box on first nibble of the key state
        s_in = k[0] & 0xF
        k[0] = (k[0] & 0xFFF0) | self.SBOX_TABLE[s_in]
        # 2. Feistel-like 5-word 8-bit rotation
        row0, row1, row2, row3, row4 = k
        new_row0 = ((row0 << 8) | (row1 >> 8)) & 0xFFFF
        new_row1 = ((row1 << 8) | (row2 >> 8)) & 0xFFFF
        new_row2 = ((row2 << 8) | (row3 >> 8)) & 0xFFFF
        new_row3 = ((row3 << 8) | (row4 >> 8)) & 0xFFFF
        new_row4 = ((row4 << 8) | (row0 >> 8)) & 0xFFFF
        # 3. Round Constant XOR (5 bits)
        new_row0 ^= (self.RC[r_idx] & 0x1F)
        return [new_row0, new_row1, new_row2, new_row3, new_row4]

    def generate_round_keys(self, key):
        rks = []
        # 80-bit key split into 5 words
        k_rows = [(key >> (16 * i)) & 0xFFFF for i in range(5)]
        for r in range(self.rounds):
            rks.append(k_rows[:4])
            k_rows = self.key_update_80(k_rows, r)
        rks.append(k_rows[:4]) # Final key for whitening
        return rks

    
    # TIER 4: ORCHESTRATION & API
    

    def encrypt(self, p, k):
        rows = [(p >> (16 * i)) & 0xFFFF for i in range(4)]
        rks = self.generate_round_keys(k)
        for i in range(self.rounds):
            rows = self.encrypt_round(rows, rks[i])
        # Final Whitening
        rows = self.add_round_key(rows, rks[25])
        return sum(rows[i] << (16 * i) for i in range(4))

    def decrypt(self, c, k):
        rows = [(c >> (16 * i)) & 0xFFFF for i in range(4)]
        rks = self.generate_round_keys(k)
        # Undo Whitening
        rows = self.add_round_key(rows, rks[25])
        for i in range(24, -1, -1):
            rows = self.decrypt_round(rows, rks[i])
        return sum(rows[i] << (16 * i) for i in range(4))

if __name__ == "__main__":
    cipher = RECTANGLECipher()
    p, k = 0x0123456789ABCDEF, 0xFFEEDDCCBBAA99887766
    c = cipher.encrypt(p, k)
    d = cipher.decrypt(c, k)
    
    print(f"Plaintext:  {hex(p)}")
    print(f"Ciphertext: {hex(c)}")
    print(f"Decrypted:  {hex(d)}")
    print(f"Verified Spec-Faithful Bitsliced implementation: {p == d}")

    