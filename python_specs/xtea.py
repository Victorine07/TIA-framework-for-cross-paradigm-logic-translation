

class XteaCipher:
    def __init__(self, key, cycles=32):
        """
        XTEA-64/128
        :param key: 128-bit master key (int)
        :param cycles: 32 cycles (equivalent to 64 Feistel rounds)
        """
        self.cycles = cycles
        self.delta = 0x9E3779B9
        self.mask = 0xFFFFFFFF
        
        # Key Schedule: 128-bit key split into four 32-bit words
        # Consistent with standard big-endian interpretation
        self.key_words = [
            (key >> 96) & self.mask,
            (key >> 64) & self.mask,
            (key >> 32) & self.mask,
            key & self.mask
        ]

    def encrypt_round(self, v0, v1, sum_val):
        """
        Single XTEA Cycle (2 Half-Rounds)
        Matches logic in ArxPy/primitives/xtea.py
        """
        # 1. Update V0 using the current sum
        k0 = self.key_words[sum_val & 3]
        v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum_val + k0))) & self.mask
        
        # 2. Update Sum (The Delta constant)
        new_sum = (sum_val + self.delta) & self.mask
        
        # 3. Update V1 using the updated sum
        k1 = self.key_words[(new_sum >> 11) & 3]
        v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (new_sum + k1))) & self.mask
        
        return v0, v1, new_sum

    def decrypt_round(self, v0, v1, sum_val):
        """
        Inverse XTEA Cycle (2 Half-Rounds)
        """
        # 1. Reverse V1 update (uses the sum at the end of the round)
        k1 = self.key_words[(sum_val >> 11) & 3]
        v1 = (v1 - ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum_val + k1))) & self.mask
        
        # 2. Reverse Sum update
        prev_sum = (sum_val - self.delta) & self.mask
        
        # 3. Reverse V0 update (uses the restored sum)
        k0 = self.key_words[prev_sum & 3]
        v0 = (v0 - ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (prev_sum + k0))) & self.mask
        
        return v0, v1, prev_sum

    def encrypt(self, plaintext):
        """Full XTEA Encryption (ECB)"""
        v0 = (plaintext >> 32) & self.mask
        v1 = plaintext & self.mask
        sum_val = 0
        
        for _ in range(self.cycles):
            v0, v1, sum_val = self.encrypt_round(v0, v1, sum_val)
            
        return (v0 << 32) | v1

    def decrypt(self, ciphertext):
        """Full XTEA Decryption (ECB)"""
        v0 = (ciphertext >> 32) & self.mask
        v1 = ciphertext & self.mask
        
        # sum = delta * cycles (Exactly 32 increments for 32 cycles)
        sum_val = (self.delta * self.cycles) & self.mask
        
        for _ in range(self.cycles):
            v0, v1, sum_val = self.decrypt_round(v0, v1, sum_val)
            
        return (v0 << 32) | v1

def test_xtea_vectors():
    """
    Official Test Vector Verification
    Key: 000102030405060708090a0b0c0d0e0f
    Plain: 4142434445464748
    Expected Cipher: 497DF3D072612CB5
    """
    key = 0x000102030405060708090A0B0C0D0E0F
    plain = 0x4142434445464748
    expected = 0x497DF3D072612CB5
    
    cipher = XteaCipher(key)
    res_enc = cipher.encrypt(plain)
    res_dec = cipher.decrypt(res_enc)
    
    print(f"Test Vector Check:")
    print(f"  Result: {res_enc:016X}")
    print(f"  Target: {expected:016X}")
    print(f"  Status: {'PASSED' if res_enc == expected else 'FAILED'}")
    print(f"  Decryption Integrity: {'PASSED' if res_dec == plain else 'FAILED'}")

if __name__ == "__main__":
    test_xtea_vectors()