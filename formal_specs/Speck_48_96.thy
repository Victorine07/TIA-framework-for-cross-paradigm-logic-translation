theory Speck_48_96
  imports
    "HOL-Library.Word"
    "HOL.Bit_Operations"
begin


definition speck_48_96_alpha :: nat where "speck_48_96_alpha = 8"


definition speck_48_96_beta :: nat where "speck_48_96_beta = 3"


definition speck_48_96_encrypt_round :: "24 word ⇒ 24 word × 24 word ⇒ 24 word × 24 word" where
  "speck_48_96_encrypt_round k xy = (
    let (x, y) = xy;
        rs_x = word_rotr speck_48_96_alpha x;
        add_xy = rs_x + y;
        new_x = xor add_xy k;
        ls_y = word_rotl speck_48_96_beta y;
        new_y = xor new_x ls_y
    in (new_x, new_y))"


definition speck_48_96_decrypt_round_inverse :: "24 word ⇒ 24 word × 24 word ⇒ 24 word × 24 word" where
  "speck_48_96_decrypt_round_inverse k xy_new = (
    let (x, y) = xy_new;
        xor_xy = xor x y;
        new_y = word_rotr speck_48_96_beta xor_xy;
        xor_xk = xor x k;
        msub = xor_xk - new_y;
        new_x = word_rotl speck_48_96_alpha msub
    in (new_x, new_y))"


function speck_48_96_gen_key_schedule_rec :: "24 word list ⇒ 24 word list ⇒ nat ⇒ 24 word list" where
  "speck_48_96_gen_key_schedule_rec l_keys k_keys i = (
     if i ≥ (23 - 1) then k_keys
     else
       let (new_l, new_k) = speck_48_96_encrypt_round (word_of_nat i) (l_keys ! i, k_keys ! i)
       in speck_48_96_gen_key_schedule_rec (l_keys @ [new_l]) (k_keys @ [new_k]) (i + 1))"
  by pat_completeness auto
termination by (relation "measure (λ(l, k, i). 22 - i)") auto


definition speck_48_96_generate_key_schedule :: "24 word list ⇒ 24 word list" where
  "speck_48_96_generate_key_schedule initial_key_words = (
     let k0 = [initial_key_words ! 0];
         l0 = [initial_key_words ! 1, initial_key_words ! 2, initial_key_words ! 3]
     in speck_48_96_gen_key_schedule_rec l0 k0 0)"


fun speck_48_96_encrypt_iterate :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "speck_48_96_encrypt_iterate state [] = state"
| "speck_48_96_encrypt_iterate state (k#ks) = speck_48_96_encrypt_iterate (speck_48_96_encrypt_round k state) ks"


fun speck_48_96_decrypt_iterate :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "speck_48_96_decrypt_iterate state ks = foldl (λst_new k. speck_48_96_decrypt_round_inverse k st_new) state (rev ks)"


definition speck_48_96_encrypt_block :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "speck_48_96_encrypt_block state keys = speck_48_96_encrypt_iterate state keys"


definition speck_48_96_decrypt_block :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "speck_48_96_decrypt_block state keys = speck_48_96_decrypt_iterate state keys"


definition speck_48_96_encrypt :: "48 word ⇒ 24 word list ⇒ 48 word" where
  "speck_48_96_encrypt plaintext keys = (
    let left = ucast (drop_bit 24 plaintext);
        right = ucast plaintext;
        (c_l, c_r) = speck_48_96_encrypt_block (left, right) keys
    in or (push_bit 24 (ucast c_l)) (ucast c_r))"


definition speck_48_96_decrypt :: "48 word ⇒ 24 word list ⇒ 48 word" where
  "speck_48_96_decrypt ciphertext keys = (
    let left = ucast (drop_bit 24 ciphertext);
        right = ucast ciphertext;
        (p_l, p_r) = speck_48_96_decrypt_block (left, right) keys
    in or (push_bit 24 (ucast p_l)) (ucast p_r))"


end