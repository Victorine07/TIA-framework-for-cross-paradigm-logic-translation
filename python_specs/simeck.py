

NUM_ROUNDS = {
    (32, 64): 32,
    (48, 96): 36,
    (64, 128): 44,
}

def get_sequence(num_rounds):
    """Generate LFSR sequence for round constants"""
    if num_rounds < 40:
        states = [1] * 5
    else:
        states = [1] * 6

    for i in range(num_rounds - 5):
        if num_rounds < 40:
            feedback = states[i + 2] ^ states[i]
        else:
            feedback = states[i + 1] ^ states[i]
        states.append(feedback)

    return tuple(states)

class Simeck:
    def __init__(self, block_size, key_size, master_key):
        assert (block_size, key_size) in NUM_ROUNDS
        assert 0 <= master_key < (1 << key_size)
        self._block_size = block_size
        self._key_size = key_size
        self._word_size = block_size // 2
        self._num_rounds = NUM_ROUNDS[(block_size, key_size)]
        self._sequence = get_sequence(self._num_rounds)
        self._mask = (1 << self._word_size) - 1  # Use mask, not modulus
        self.change_key(master_key)

    def _LROT(self, x, r):
        """Left rotation using | & mask pattern (consistent with Simon)"""
        return ((x << r) | (x >> (self._word_size - r))) & self._mask

    def _round(self, round_key, left, right):
        """Simeck round function - using | & mask pattern"""
        # Use consistent rotation pattern
        rot5 = self._LROT(left, 5)
        rot1 = self._LROT(left, 1)
        
        temp = left
        left = right ^ (left & rot5) ^ rot1 ^ round_key
        right = temp
        
        return left, right

    def change_key(self, master_key):
        """Key schedule - using | & mask pattern consistently"""
        states = []
        for i in range(self._key_size // self._word_size):
            states.append(master_key & self._mask)
            master_key >>= self._word_size

        constant = self._mask ^ 3  # 0xFF...FC (using mask)
        round_keys = []
        
        for i in range(self._num_rounds):
            round_keys.append(states[0])
            left, right = states[1], states[0]
            left, right = self._round(constant ^ self._sequence[i], left, right)
            states.append(left)
            states.pop(0)
            states[0] = right

        self.__round_keys = tuple(round_keys)

    def encrypt(self, plaintext):
        """Encryption - same as before but using corrected pattern"""
        left = plaintext >> self._word_size
        right = plaintext & self._mask

        for idx in range(self._num_rounds):
            left, right = self._round(self.__round_keys[idx], left, right)

        return (left << self._word_size) | right

    def decrypt(self, ciphertext):
        """Decryption - using | & mask pattern consistently"""
        left = ciphertext >> self._word_size
        right = ciphertext & self._mask
        
        # Decrypt by applying rounds in reverse order
        for idx in range(self._num_rounds - 1, -1, -1):
            # For decryption: right is the F-function input
            rot5 = self._LROT(right, 5)
            rot1 = self._LROT(right, 1)
            f_val = (right & rot5) ^ rot1
            
            # Inverse Feistel
            temp = left
            left = right
            right = temp ^ f_val ^ self.__round_keys[idx]
        
        return (left << self._word_size) | right


# Test the fixed implementation
if __name__ == "__main__":
    # Test vector from original Simeck
    plaintext32 = 0x65656877
    key64 = 0x1918111009080100
    simeck32 = SimeckFixed(32, 64, key64)
    ciphertext32 = simeck32.encrypt(plaintext32)
    
    print(f"Plaintext:  {plaintext32:08x}")
    print(f"Ciphertext: {ciphertext32:08x}")
    
    # Test decryption
    decrypted = simeck32.decrypt(ciphertext32)
    print(f"Decrypted:  {decrypted:08x}")
    print(f"Match: {decrypted == plaintext32}")
