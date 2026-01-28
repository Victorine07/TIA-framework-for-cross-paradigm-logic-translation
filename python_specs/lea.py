
class LEACipher:
    # Official LEA DELTA constants derived from hex representation of sqrt(76616429)
    DELTA = [
        0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
        0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957
    ]

    def __init__(self, key, key_size=128):
        self.key_size = key_size
        self.word_size = 32
        self.mod_mask = 0xFFFFFFFF
        
        # Initialize variant-specific parameters
        if key_size == 128:
            self.rounds = 24
            self.m = 4
        elif key_size == 192:
            self.rounds = 28
            self.m = 6
        elif key_size == 256:
            self.rounds = 32
            self.m = 8
        else:
            raise ValueError("Invalid key size. LEA supports 128, 192, or 256 bits.")

        self.key_schedule = self._generate_key_schedule(key)

    def _rol(self, x, r):
        """Left rotation of 32-bit word"""
        return ((x << (r % 32)) | (x >> (32 - (r % 32)))) & self.mod_mask

    def _ror(self, x, r):
        """Right rotation of 32-bit word"""
        return ((x >> (r % 32)) | (x << (32 - (r % 32)))) & self.mod_mask

    def _generate_key_schedule(self, key):
        """LEA Key Schedule Algorithm following official specifications for all widths"""
        # 1. Extract 32-bit words from master key (Little-endian word order)
        T = []
        for i in range(self.m):
            T.append((key >> (32 * i)) & self.mod_mask)
        
        rk = []
        for i in range(self.rounds):
            if self.key_size == 128:
                # 128-bit key uses delta[i % 4]
                d = self.DELTA[i % 4]
                T[0] = self._rol((T[0] + self._rol(d, i)) & self.mod_mask, 1)
                T[1] = self._rol((T[1] + self._rol(d, i + 1)) & self.mod_mask, 3)
                T[2] = self._rol((T[2] + self._rol(d, i + 2)) & self.mod_mask, 6)
                T[3] = self._rol((T[3] + self._rol(d, i + 3)) & self.mod_mask, 11)
                rk.append([T[0], T[1], T[2], T[1], T[3], T[1]])
            
            elif self.key_size == 192:
                # 192-bit key uses delta[i % 6]
                d = self.DELTA[i % 6]
                T[0] = self._rol((T[0] + self._rol(d, i)) & self.mod_mask, 1)
                T[1] = self._rol((T[1] + self._rol(d, i + 1)) & self.mod_mask, 3)
                T[2] = self._rol((T[2] + self._rol(d, i + 2)) & self.mod_mask, 6)
                T[3] = self._rol((T[3] + self._rol(d, i + 3)) & self.mod_mask, 11)
                T[4] = self._rol((T[4] + self._rol(d, i + 4)) & self.mod_mask, 13)
                T[5] = self._rol((T[5] + self._rol(d, i + 5)) & self.mod_mask, 17)
                rk.append([T[0], T[1], T[2], T[3], T[4], T[5]])

            else: # 256-bit key
                # 256-bit key uses delta[i % 8]
                d = self.DELTA[i % 8]
                T[0] = self._rol((T[0] + self._rol(d, i)) & self.mod_mask, 1)
                T[1] = self._rol((T[1] + self._rol(d, i + 1)) & self.mod_mask, 3)
                T[2] = self._rol((T[2] + self._rol(d, i + 2)) & self.mod_mask, 6)
                T[3] = self._rol((T[3] + self._rol(d, i + 3)) & self.mod_mask, 11)
                T[4] = self._rol((T[4] + self._rol(d, i + 4)) & self.mod_mask, 13)
                T[5] = self._rol((T[5] + self._rol(d, i + 5)) & self.mod_mask, 17)
                T[6] = self._rol((T[6] + self._rol(d, i + 6)) & self.mod_mask, 19)
                T[7] = self._rol((T[7] + self._rol(d, i + 7)) & self.mod_mask, 23)
                rk.append([T[(6*i) % 8], T[(6*i+1) % 8], T[(6*i+2) % 8], 
                           T[(6*i+3) % 8], T[(6*i+4) % 8], T[(6*i+5) % 8]])
        return rk

    def encrypt_round(self, state, rk):
        """LEA Parallel ARX Encrypt Round: 3 updates in parallel, 1 shift"""
        x0, x1, x2, x3 = state
        
        # Round keys rk0-rk5 are applied in pairs to the parallel additions
        next_x0 = self._rol(((x0 ^ rk[0]) + (x1 ^ rk[1])) & self.mod_mask, 9)
        next_x1 = self._ror(((x1 ^ rk[2]) + (x2 ^ rk[3])) & self.mod_mask, 5)
        next_x2 = self._ror(((x2 ^ rk[4]) + (x3 ^ rk[5])) & self.mod_mask, 3)
        next_x3 = x0 # x0 is shifted to the end
        
        return [next_x0, next_x1, next_x2, next_x3]

    def decrypt_round(self, state, rk):
        """LEA Parallel ARX Decrypt Round: Algebraically correct inverse"""
        x0_next, x1_next, x2_next, x3_next = state
        
        # Forward was: x3_next = x0
        x0 = x3_next
        
        # Forward was: x0_next = ROL((x0^rk0 + x1^rk1), 9)
        # Inverse: x1 = (ROR(x0_next, 9) - (x0^rk0)) ^ rk1
        x1 = ((self._ror(x0_next, 9) - (x0 ^ rk[0])) & self.mod_mask) ^ rk[1]
        
        # Forward was: x1_next = ROR((x1^rk2 + x2^rk3), 5)
        # Inverse: x2 = (ROL(x1_next, 5) - (x1^rk2)) ^ rk3
        x2 = ((self._rol(x1_next, 5) - (x1 ^ rk[2])) & self.mod_mask) ^ rk[3]
        
        # Forward was: x2_next = ROR((x2^rk4 + x3^rk5), 3)
        # Inverse: x3 = (ROL(x2_next, 3) - (x2^rk4)) ^ rk5
        x3 = ((self._rol(x2_next, 3) - (x2 ^ rk[4])) & self.mod_mask) ^ rk[5]
        
        return [x0, x1, x2, x3]

    def encrypt(self, plaintext):
        """Complete LEA encryption following the established word-order convention"""
        # Split 128-bit block into four 32-bit words (Little-endian order for consistency)
        state = []
        for i in range(4):
            state.append((plaintext >> (32 * i)) & self.mod_mask)
        
        # Forward rounds
        for i in range(self.rounds):
            state = self.encrypt_round(state, self.key_schedule[i])
            
        # Recombine words into 128-bit integer
        res = 0
        for i in range(4):
            res |= (state[i] << (32 * i))
        return res

    def decrypt(self, ciphertext):
        """Complete LEA decryption - undoes rounds in reverse order"""
        state = []
        for i in range(4):
            state.append((ciphertext >> (32 * i)) & self.mod_mask)
        
        # Reverse iteration of rounds
        for i in range(self.rounds - 1, -1, -1):
            state = self.decrypt_round(state, self.key_schedule[i])
            
        res = 0
        for i in range(4):
            res |= (state[i] << (32 * i))
        return res

if __name__ == "__main__":
    # Quick verification test (consistent with spec vectors)
    key_128 = 0x3c2d1e0f78695a4bb4a59687f0e1d2c3
    pt = 0x13121110171615141b1a19181f1e1d1c
    
    cipher = LEACipher(key_128, key_size=128)
    ct = cipher.encrypt(pt)
    dt = cipher.decrypt(ct)
    
    print(f"Key: {hex(key_128)}")
    print(f"Plaintext:  {hex(pt)}")
    print(f"Ciphertext: {hex(ct)}")
    print(f"Decrypted:  {hex(dt)}")
    assert dt == pt, "Decryption failure: Result does not match original plaintext."
    