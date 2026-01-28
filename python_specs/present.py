

class PRESENTCipher:
    """
    PRESENT Block Cipher - Cryptographically correct implementation.
    Standardized for LLM logic induction and formal verification.
    """

    # Official PRESENT S-Box
    SBOX = [
        0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 
        0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    ]

    # Verified inverse S-Box
    INV_SBOX = [
        0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 
        0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA
    ]

    # Bit Permutation Table P(i)
    PERM = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
    ]

    # Inverse Permutation Table
    INV_PERM = [0] * 64

    def __init__(self, key_size=80):
        self.block_size = 64
        self.key_size = key_size
        
        # Standard variants use 31 rounds; 40-bit weak variant uses 9
        if key_size == 40:
            self.rounds = 9
        else:
            self.rounds = 31

        # Precompute inverse permutation for decryption
        for i in range(64):
            self.INV_PERM[self.PERM[i]] = i

    # =========================================================================
    # CORE ROUND PRIMITIVES (Level 1)
    # =========================================================================

    def add_round_key(self, state, round_key):
        """XOR the state with the 64-bit round key"""
        return state ^ round_key

    def sbox_layer(self, state):
        """Apply 4-bit S-Box to 16 parallel nibbles"""
        res = 0
        for i in range(16):
            nibble = (state >> (4 * i)) & 0xF
            res |= (self.SBOX[nibble] << (4 * i))
        return res

    def inv_sbox_layer(self, state):
        """Inverse S-box layer for decryption"""
        res = 0
        for i in range(16):
            nibble = (state >> (4 * i)) & 0xF
            res |= (self.INV_SBOX[nibble] << (4 * i))
        return res

    def p_layer(self, state):
        """Bit-level permutation layer"""
        res = 0
        for i in range(64):
            bit = (state >> i) & 1
            res |= (bit << self.PERM[i])
        return res

    def inv_p_layer(self, state):
        """Inverse bit-level permutation layer"""
        res = 0
        for i in range(64):
            bit = (state >> i) & 1
            res |= (bit << self.INV_PERM[i])
        return res

    def encrypt_round(self, state, round_key):
        """Standard SPN round composition"""
        state = self.add_round_key(state, round_key)
        state = self.sbox_layer(state)
        state = self.p_layer(state)
        return state

    def decrypt_round(self, state, round_key):
        """Inverse SPN round composition"""
        state = self.inv_p_layer(state)
        state = self.inv_sbox_layer(state)
        state = self.add_round_key(state, round_key)
        return state

    # =========================================================================
    # KEY SCHEDULE LOGIC
    # =========================================================================

    def key_update_40(self, key_state, round_counter):
        """Update 40-bit key state (Weak variant)"""
        # 1. Rotate 20 bits
        key_state = ((key_state << 20) & ((1 << 40) - 1)) | (key_state >> 20)
        # 2. S-Box the top nibble
        nibble = (key_state >> 36) & 0xF
        key_state = (self.SBOX[nibble] << 36) | (key_state & ((1 << 36) - 1))
        # 3. XOR round counter
        key_state ^= round_counter
        return key_state

    def key_update_80(self, key_state, round_counter):
        """Update 80-bit key state (Standard)"""
        key_state = ((key_state << 61) & ((1 << 80) - 1)) | (key_state >> 19)
        ms_nibble = (key_state >> 76) & 0xF
        key_state = (self.SBOX[ms_nibble] << 76) | (key_state & ((1 << 76) - 1))
        key_state ^= (round_counter << 15)
        return key_state

    def key_update_128(self, key_state, round_counter):
        """Update 128-bit key state (Standard)"""
        key_state = ((key_state << 61) & ((1 << 128) - 1)) | (key_state >> 67)
        n1 = (key_state >> 124) & 0xF
        n2 = (key_state >> 120) & 0xF
        key_state = (self.SBOX[n1] << 124) | (self.SBOX[n2] << 120) | (key_state & ((1 << 120) - 1))
        key_state ^= (round_counter << 62)
        return key_state

    def generate_round_keys(self, key):
        """Full Key Schedule generation"""
        round_keys = []
        k_state = key
        for r in range(1, self.rounds + 2):
            if self.key_size == 40:
                round_keys.append(k_state >> 0) # Use full word for 40-bit
                k_state = self.key_update_40(k_state, r)
            elif self.key_size == 80:
                round_keys.append(k_state >> 16)
                k_state = self.key_update_80(k_state, r)
            else:
                round_keys.append(k_state >> 64)
                k_state = self.key_update_128(k_state, r)
        return round_keys

    # =========================================================================
    # ITERATION AND ORCHESTRATION (Level 2 & 3)
    # =========================================================================

    def encrypt_iterate(self, state, round_keys):
        """Recursive/Iterative layer for round application"""
        for rk in round_keys[:-1]:
            state = self.encrypt_round(state, rk)
        # Final whitening XOR
        return state ^ round_keys[-1]

    def decrypt_iterate(self, state, round_keys):
        """Recursive/Iterative layer for inverse round application"""
        # Undo final whitening
        state = state ^ round_keys[-1]
        # Undo rounds in reverse
        for rk in reversed(round_keys[:-1]):
            state = self.decrypt_round(state, rk)
        return state

    def encrypt_block(self, plaintext, key):
        """Orchestration of key expansion and state transformation"""
        round_keys = self.generate_round_keys(key)
        return self.encrypt_iterate(plaintext, round_keys)

    def decrypt_block(self, ciphertext, key):
        """Orchestration of key expansion and inverse state transformation"""
        round_keys = self.generate_round_keys(key)
        return self.decrypt_iterate(ciphertext, round_keys)

    # =========================================================================
    # TOP-LEVEL API (Level 4)
    # =========================================================================

    def encrypt(self, plaintext, key):
        """Top-level encryption entry point"""
        return self.encrypt_block(plaintext, key)

    def decrypt(self, ciphertext, key):
        """Top-level decryption entry point"""
        return self.decrypt_block(ciphertext, key)

# Verification block
if __name__ == "__main__":
    # Standard Test: 80-bit All Zeros
    cipher80 = PRESENTCipher(80)
    c80 = cipher80.encrypt(0x0, 0x0)
    print(f"PRESENT-80 All Zeros Pass: {c80 == 0x5579C1387B228445}")
    
    # Standard Test: 128-bit All Zeros
    cipher128 = PRESENTCipher(128)
    c128 = cipher128.encrypt(0x0, 0x0)
    print(f"PRESENT-128 All Zeros Pass: {c128 == 0x96db702a2e6900af}")

    # Weak Variant Test: 40-bit
    cipher40 = PRESENTCipher(40)
    c40 = cipher40.encrypt(0x0, 0x0)
    print(f"PRESENT-40 (9 rounds) Result: {hex(c40)}")
    print(f"PRESENT-40 Decrypt Pass: {cipher40.decrypt(c40, 0x0) == 0x0}")