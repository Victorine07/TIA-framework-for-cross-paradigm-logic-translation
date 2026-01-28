theory Speck_96_144
  imports
    "HOL-Library.Word"
    "HOL.Bit_Operations"
begin

definition speck_96_144_alpha :: nat where "speck_96_144_alpha = 8"


definition speck_96_144_beta :: nat where "speck_96_144_beta = 3"


definition speck_96_144_encrypt_round :: "48 word ⇒ 48 word × 48 word ⇒ 48 word × 48 word" where
  "speck_96_144_encrypt_round k xy = (
    let (x, y) = xy;
        rs_x = word_rotr speck_96_144_alpha x;
        add_xy = rs_x + y;
        new_x = xor add_xy k;
        ls_y = word_rotl speck_96_144_beta y;
        new_y = xor new_x ls_y
    in (new_x, new_y))"


definition speck_96_144_decrypt_round_inverse :: "48 word ⇒ 48 word × 48 word ⇒ 48 word × 48 word" where
  "speck_96_144_decrypt_round_inverse k xy_new = (
    let (x, y) = xy_new;
        xor_xy = xor x y;
        new_y = word_rotr speck_96_144_beta xor_xy;
        xor_xk = xor x k;
        msub = xor_xk - new_y;
        new_x = word_rotl speck_96_144_alpha msub
    in (new_x, new_y))"


function speck_96_144_gen_key_schedule_rec :: "48 word list ⇒ 48 word list ⇒ nat ⇒ 48 word list" where
  "speck_96_144_gen_key_schedule_rec l_keys k_keys i = (
     if i ≥ (29 - 1) then k_keys
     else
       let (new_l, new_k) = speck_96_144_encrypt_round (word_of_nat i) (l_keys ! i, k_keys ! i)
       in speck_96_144_gen_key_schedule_rec (l_keys @ [new_l]) (k_keys @ [new_k]) (i + 1))"
  by pat_completeness auto
termination by (relation "measure (λ(l, k, i). 28 - i)") auto


definition speck_96_144_generate_key_schedule :: "48 word list ⇒ 48 word list" where
  "speck_96_144_generate_key_schedule initial_key_words = (
     let k0 = [initial_key_words ! 0];
         l0 = [initial_key_words ! 1, initial_key_words ! 2]
     in speck_96_144_gen_key_schedule_rec l0 k0 0)"


fun speck_96_144_encrypt_iterate :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "speck_96_144_encrypt_iterate state [] = state"
| "speck_96_144_encrypt_iterate state (k#ks) = speck_96_144_encrypt_iterate (speck_96_144_encrypt_round k state) ks"


fun speck_96_144_decrypt_iterate :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "speck_96_144_decrypt_iterate state ks = foldl (λst_new k. speck_96_144_decrypt_round_inverse k st_new) state (rev ks)"


definition speck_96_144_encrypt_block :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "speck_96_144_encrypt_block state keys = speck_96_144_encrypt_iterate state keys"


definition speck_96_144_decrypt_block :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "speck_96_144_decrypt_block state keys = speck_96_144_decrypt_iterate state keys"


definition speck_96_144_encrypt :: "96 word ⇒ 48 word list ⇒ 96 word" where
  "speck_96_144_encrypt plaintext keys = (
    let left = ucast (drop_bit 48 plaintext);
        right = ucast plaintext;
        (c_l, c_r) = speck_96_144_encrypt_block (left, right) keys
    in or (push_bit 48 (ucast c_l)) (ucast c_r))"


definition speck_96_144_decrypt :: "96 word ⇒ 48 word list ⇒ 96 word" where
  "speck_96_144_decrypt ciphertext keys = (
    let left = ucast (drop_bit 48 ciphertext);
        right = ucast ciphertext;
        (p_l, p_r) = speck_96_144_decrypt_block (left, right) keys
    in or (push_bit 48 (ucast p_l)) (ucast p_r))"


end