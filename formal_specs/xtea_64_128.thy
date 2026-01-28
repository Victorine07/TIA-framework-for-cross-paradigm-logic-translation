theory xtea_64_128
  imports
    "HOL-Library.Word"
    "HOL.Bit_Operations"
begin

definition xtea_64_128_delta :: "32 word" where
  "xtea_64_128_delta = 0x9E3779B9"


definition xtea_64_128_num_cycles :: nat where
  "xtea_64_128_num_cycles = 32"


definition xtea_64_128_split_key :: "128 word \<Rightarrow> 32 word list" where
  "xtea_64_128_split_key key = [
    ucast (drop_bit 96 key),
    ucast (drop_bit 64 key),
    ucast (drop_bit 32 key),
    ucast key
  ]"


definition xtea_64_128_mix_function :: "32 word \<Rightarrow> 32 word" where
  "xtea_64_128_mix_function x = xor (push_bit 4 x) (drop_bit 5 x)"


definition xtea_64_128_encrypt_round :: 
  "32 word list \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> (32 word \<times> 32 word \<times> 32 word)" where
  "xtea_64_128_encrypt_round key_words v0 v1 s_val = (
    let k0 = key_words ! (unat (and s_val 3));
        mix_v1 = xtea_64_128_mix_function v1;
        v0_new = v0 + (xor (mix_v1 + v1) (s_val + k0));
        s_new = s_val + xtea_64_128_delta;
        k1 = key_words ! (unat (and (drop_bit 11 s_new) 3));
        mix_v0_new = xtea_64_128_mix_function v0_new;
        v1_new = v1 + (xor (mix_v0_new + v0_new) (s_new + k1))
    in (v0_new, v1_new, s_new))"


definition xtea_64_128_decrypt_round :: 
  "32 word list \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> (32 word \<times> 32 word \<times> 32 word)" where
  "xtea_64_128_decrypt_round key_words v0 v1 s_val = (
    let k1 = key_words ! (unat (and (drop_bit 11 s_val) 3));
        mix_v0 = xtea_64_128_mix_function v0;
        v1_prev = v1 - (xor (mix_v0 + v0) (s_val + k1));
        s_prev = s_val - xtea_64_128_delta;
        k0 = key_words ! (unat (and s_prev 3));
        mix_v1_prev = xtea_64_128_mix_function v1_prev;
        v0_prev = v0 - (xor (mix_v1_prev + v1_prev) (s_prev + k0))
    in (v0_prev, v1_prev, s_prev))"


fun xtea_64_128_encrypt_iterate :: 
  "32 word list \<Rightarrow> nat \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> (32 word \<times> 32 word)" where
  "xtea_64_128_encrypt_iterate key_words 0 v0 v1 s = (v0, v1)"
| "xtea_64_128_encrypt_iterate key_words (Suc n) v0 v1 s = (
    let (v0', v1', s') = xtea_64_128_encrypt_round key_words v0 v1 s
    in xtea_64_128_encrypt_iterate key_words n v0' v1' s')"


fun xtea_64_128_decrypt_iterate :: 
  "32 word list \<Rightarrow> nat \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> 32 word \<Rightarrow> (32 word \<times> 32 word)" where
  "xtea_64_128_decrypt_iterate key_words 0 v0 v1 s = (v0, v1)"
| "xtea_64_128_decrypt_iterate key_words (Suc n) v0 v1 s = (
    let (v0', v1', s') = xtea_64_128_decrypt_round key_words v0 v1 s
    in xtea_64_128_decrypt_iterate key_words n v0' v1' s')"


definition xtea_64_128_encrypt_block :: 
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
  "xtea_64_128_encrypt_block key plaintext = (
    let key_words = xtea_64_128_split_key key;
        v0 = ucast (drop_bit 32 plaintext);
        v1 = ucast plaintext;
        (v0_final, v1_final) = xtea_64_128_encrypt_iterate key_words xtea_64_128_num_cycles v0 v1 0
    in or (push_bit 32 (ucast v0_final)) (ucast v1_final))"


definition xtea_64_128_decrypt_block :: 
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
  "xtea_64_128_decrypt_block key ciphertext = (
    let key_words = xtea_64_128_split_key key;
        v0 = ucast (drop_bit 32 ciphertext);
        v1 = ucast ciphertext;
        initial_sum = xtea_64_128_delta * (word_of_nat xtea_64_128_num_cycles);
        (v0_final, v1_final) = xtea_64_128_decrypt_iterate key_words xtea_64_128_num_cycles v0 v1 initial_sum
    in or (push_bit 32 (ucast v0_final)) (ucast v1_final))"



definition xtea_64_128_encrypt :: 
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
  "xtea_64_128_encrypt key plaintext = 
    xtea_64_128_encrypt_block key plaintext"


definition xtea_64_128_decrypt :: 
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
  "xtea_64_128_decrypt key ciphertext = 
    xtea_64_128_decrypt_block key ciphertext"


definition xtea_64_128_test_key :: "128 word" where
  "xtea_64_128_test_key = 0x000102030405060708090A0B0C0D0E0F"


definition xtea_64_128_test_plaintext :: "64 word" where
  "xtea_64_128_test_plaintext = 0x4142434445464748"


definition xtea_64_128_expected_ciphertext :: "64 word" where
  "xtea_64_128_expected_ciphertext = 0x497DF3D072612CB5"


definition xtea_64_128_actual_ciphertext :: "64 word" where
  "xtea_64_128_actual_ciphertext = xtea_64_128_encrypt_block xtea_64_128_test_key xtea_64_128_test_plaintext"


definition xtea_64_128_decrypted_plaintext :: "64 word" where
  "xtea_64_128_decrypted_plaintext = xtea_64_128_decrypt_block xtea_64_128_test_key xtea_64_128_actual_ciphertext"


value "xtea_64_128_actual_ciphertext"

value "xtea_64_128_decrypted_plaintext"

lemma xtea_64_128_test_vector_verification:
  "xtea_64_128_actual_ciphertext = xtea_64_128_expected_ciphertext"
  unfolding xtea_64_128_actual_ciphertext_def xtea_64_128_test_plaintext_def xtea_64_128_test_key_def 
            xtea_64_128_expected_ciphertext_def xtea_64_128_encrypt_block_def
            xtea_64_128_split_key_def xtea_64_128_delta_def xtea_64_128_num_cycles_def
            xtea_64_128_encrypt_round_def xtea_64_128_mix_function_def
  by eval

lemma xtea_64_128_decryption_integrity:
  "xtea_64_128_decrypted_plaintext = xtea_64_128_test_plaintext"
  unfolding xtea_64_128_decrypted_plaintext_def xtea_64_128_actual_ciphertext_def 
            xtea_64_128_test_plaintext_def xtea_64_128_test_key_def xtea_64_128_decrypt_block_def
            xtea_64_128_encrypt_block_def xtea_64_128_split_key_def xtea_64_128_delta_def 
            xtea_64_128_num_cycles_def xtea_64_128_encrypt_round_def xtea_64_128_decrypt_round_def
            xtea_64_128_mix_function_def
  by eval


end