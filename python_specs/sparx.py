

class SparxCipher:
    """
    Parametric SPARX Block Cipher Object.
    
    Implementation Notes:
    - SPARX ALWAYS uses 16-bit words.
    - Each branch consists of 2 × 16-bit words.
    - SPARX-64: 2 branches → 4 words → 64-bit block.
    - SPARX-128: 4 branches → 8 words → 128-bit block.
    - Key schedule operates exclusively on 16-bit words.
    - Endianness: Little-endian at the 16-bit word level.
    """

    # (block_size, key_size, n_steps, rounds_per_step, n_branches  )
    __variants = {
        'S-64/128':  (64, 128, 8, 3, 2),
        'S-128/128': (128, 128, 8, 4, 4),
        'S-128/256': (128, 256, 10, 4, 4)
    }

    def __init__(self, key, variant='S-64/128'):
        if variant not in self.__variants:
            raise ValueError(f"Invalid variant: {variant}")
            
        self.block_size, self.key_size, self.n_steps, self.rps, self.n_branches = self.__variants[variant]
        self.word_size = 16 
        self.n_words = 2 * self.n_branches
        self.mask = 0xFFFF 
        
        self.master_key = key
        self.total_rounds = self.n_steps * self.rps
        
        # Key schedule produces a flat list of 16-bit words
        self.key_words = self._generate_key_schedule()

     
    # Bitwise Helpers
     
    def _rol(self, x, r, width):
        mask = (1 << width) - 1
        return ((x << r) | (x >> (width - r))) & mask

    def _ror(self, x, r, width):
        mask = (1 << width) - 1
        return ((x >> r) | (x << (width - r))) & mask

     
    # A-permutations
     
    def A(self, x, y, width):
        """ARX A-permutation. Parametric by width (16 or 32)."""
        mask = (1 << width) - 1
        x = self._ror(x, 7, width)
        x = (x + y) & mask
        y = self._rol(y, 2, width)
        y ^= x
        return x, y

    def A_inv(self, x, y, width):
        """Inverse A-permutation."""
        mask = (1 << width) - 1
        y = self._ror(y ^ x, 2, width)
        x = self._rol((x - y) & mask, 7, width)
        return x, y

     
    # Linear Layer
     
    def L_w(self, x):
        half = self.word_size // 2
        return ((x << half) ^ (x >> half)) & self.mask

    def linear_layer(self, s):
        if self.n_branches == 2:
            t = self.L_w(s[0] ^ s[1])
            return [s[2] ^ t, s[3] ^ t, s[0], s[1]]
        else:  # n_branches == 4
            t = self.L_w(s[0] ^ s[1] ^ s[2] ^ s[3])
            return [
                s[4] ^ t, s[5] ^ t,
                s[6] ^ t, s[7] ^ t,
                s[0], s[1], s[2], s[3]
            ]

    def linear_layer_inv(self, s):
        if self.n_branches == 2:
            t = self.L_w(s[2] ^ s[3])
            return [s[2], s[3], s[0] ^ t, s[1] ^ t]
        else:  # n_branches == 4
            t = self.L_w(s[4] ^ s[5] ^ s[6] ^ s[7])
            return [
                s[4], s[5], s[6], s[7],
                s[0] ^ t, s[1] ^ t, s[2] ^ t, s[3] ^ t
            ]

     
    # Key Schedule
     
    def _generate_key_schedule(self):
        """
        Generates subkeys using 16-bit words.
        Produces enough 16-bit words to satisfy all rounds + whitening.
        """
        k = [(self.master_key >> (16 * i)) & 0xFFFF for i in range(self.key_size // 16)]
        rk = []
        c = 1
        
        # We need enough 16-bit words for (total_rounds + 1) injections.
        # Each injection XORs the first word of each branch.
        # S-64 (b=2, w=16): 1 word per branch = 2 words per round.
        # Each injection XORs the first 16-bit word of each branch.
        # words_per_injection = n_branches

        words_per_injection = self.n_branches * (self.word_size // 16)
        total_k_words = (self.total_rounds + 1) * words_per_injection

        while len(rk) < total_k_words:
            # Current subkeys are k0, k1
            rk.extend([k[0], k[1]])
            
            # Update key state using 16-bit A-permutation
            k[0], k[1] = self.A(k[0], k[1], 16)
            
            if self.key_size == 128:
                k[2] = (k[2] + k[0]) & 0xFFFF
                k[3] = (k[3] + k[1] + c) & 0xFFFF
                k = k[2:] + k[:2]
            else: # 256-bit key
                k[2] = (k[2] + k[0]) & 0xFFFF
                k[3] = (k[3] + k[1]) & 0xFFFF
                k[4] = (k[4] + c) & 0xFFFF
                k[5] = (k[5] + (c >> 8)) & 0xFFFF
                k = k[3:] + k[:3]
            c += 1
            
        return rk

    def _get_round_key(self, r_idx):
        """Pulls the appropriate subkeys for a round and formats them to word_size."""
        n_16bit_words = self.word_size // 16
        keys = []
        base = r_idx * self.n_branches * n_16bit_words
        
        for b in range(self.n_branches):
            val = 0
            for i in range(n_16bit_words):
                # Little-endian concatenation of 16-bit words into larger word
                val |= (self.key_words[base + b * n_16bit_words + i] << (16 * i))
            keys.append(val)
        return keys

     
    # Main logic
     
    def encrypt(self, plaintext):
        # 1. Unpack plaintext into 16-bit state words
        s = self._unpack_state(plaintext)
    
        # 2. Forward rounds
        for step in range(self.n_steps):
            for r in range(self.rps):
                rk = self._get_round_key(step * self.rps + r)
                for b in range(self.n_branches):
                    s[2 * b] ^= rk[b]
                    s[2 * b], s[2 * b + 1] = self.A(
                        s[2 * b], s[2 * b + 1], 16
                    )
    
            if step < self.n_steps - 1:
                s = self.linear_layer(s)
    
        # 3. Final whitening
        wk = self._get_round_key(self.total_rounds)
        for b in range(self.n_branches):
            s[2 * b] ^= wk[b]
    
        # 4. Pack state back into integer
        return self._pack_state(s)


    def _unpack_state(self, x):
        """Unpack integer into N_WORDS little-endian 16-bit words."""
        return [(x >> (16 * i)) & 0xFFFF for i in range(self.n_words)]
    
    def _pack_state(self, s):
        """Pack N_WORDS little-endian 16-bit words into integer."""
        return sum((s[i] & 0xFFFF) << (16 * i) for i in range(self.n_words))

        
    def decrypt(self, ciphertext):
        # 1. Unpack ciphertext into 16-bit state words
        s = self._unpack_state(ciphertext)
    
        # 2. Undo final whitening
        wk = self._get_round_key(self.total_rounds)
        for b in range(self.n_branches):
            s[2 * b] ^= wk[b]
    
        # 3. Main inverse rounds
        for step in reversed(range(self.n_steps)):
    
            # Inverse linear layer (except after last step)
            if step < self.n_steps - 1:
                s = self.linear_layer_inv(s)
    
            # Inverse ARX rounds
            for r in reversed(range(self.rps)):
                rk = self._get_round_key(step * self.rps + r)
                for b in range(self.n_branches):
                    s[2 * b], s[2 * b + 1] = self.A_inv(
                        s[2 * b], s[2 * b + 1], 16
                    )
                    s[2 * b] ^= rk[b]
    
        # 4. Pack state back into integer
        return self._pack_state(s)

if __name__ == "__main__":
    test_variants = ['S-64/128', 'S-128/128', 'S-128/256']
    for v in test_variants:
        if v == 'S-128/256':
            key = 0x0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF
            p = 0x00112233445566778899AABBCCDDEEFF
        elif v == 'S-128/128':
            key = 0x0123456789ABCDEF0123456789ABCDEF
            p = 0x00112233445566778899AABBCCDDEEFF
        else: # S-64/128
            key = 0x0123456789ABCDEF0123456789ABCDEF
            p = 0x0011223344556677
            
        cipher = SparxCipher(key, variant=v)
        c = cipher.encrypt(p)
        d = cipher.decrypt(c)
        assert d == p
        print(f"{v:9} Verified (P: {hex(p)[:18]}... -> C: {hex(c)[:18]}...)")
        



