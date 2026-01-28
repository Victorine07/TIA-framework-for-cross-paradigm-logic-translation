theory Simon_48_72
  imports "HOL-Library.Word" "HOL.Bit_Operations"
begin


definition simon_48_72_z0 :: int where "simon_48_72_z0 = 0b01100111000011010100100010111110110011100001101010010001011111"


definition simon_48_72_F_function :: "24 word ⇒ 24 word" where
  "simon_48_72_F_function x = xor (and (word_rotl 1 x) (word_rotl 8 x)) (word_rotl 2 x)"


definition simon_48_72_encrypt_round :: "24 word ⇒ 24 word × 24 word ⇒ 24 word × 24 word" where
  "simon_48_72_encrypt_round k xy = (let (x, y) = xy in (xor (xor k (simon_48_72_F_function x)) y, x))"


definition simon_48_72_decrypt_round_inverse :: "24 word ⇒ 24 word × 24 word ⇒ 24 word × 24 word" where
  "simon_48_72_decrypt_round_inverse k xy_new = (let (x_new, y_new) = xy_new in (y_new, xor (xor x_new k) (simon_48_72_F_function y_new)))"


definition simon_48_72_rho_const :: "24 word" where
  "simon_48_72_rho_const = 0xFFFFFC"


function simon_48_72_gen_key_schedule_rec :: "24 word list ⇒ nat ⇒ 24 word list" where
  "simon_48_72_gen_key_schedule_rec ks i = (if i ≥ 36 then ks else
    let z_bit = bit simon_48_72_z0 (i - 3);
        rs_3 = word_rotr 3 (ks ! (i-1)); rs_1 = word_rotr 1 rs_3;
        new_k = xor (xor (xor (ks ! (i-3)) rs_3) rs_1) (xor (if z_bit then 1 else 0) simon_48_72_rho_const )
    in simon_48_72_gen_key_schedule_rec (ks @ [new_k]) (i+1))"
  by pat_completeness auto
termination by (relation "measure (λ(ks, i). 36 - i)") auto


definition simon_48_72_generate_key_schedule :: "24 word list ⇒ 24 word list" where
  "simon_48_72_generate_key_schedule init = simon_48_72_gen_key_schedule_rec init (length init)"


fun simon_48_72_encrypt_iterate :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "simon_48_72_encrypt_iterate st [] = st" | "simon_48_72_encrypt_iterate st (k#ks) = simon_48_72_encrypt_iterate (simon_48_72_encrypt_round k st) ks"


fun simon_48_72_decrypt_iterate :: "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
  "simon_48_72_decrypt_iterate st ks = foldl (λst_new k. simon_48_72_decrypt_round_inverse k st_new) st (rev ks)"



definition simon_48_72_encrypt_block ::
  "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
"simon_48_72_encrypt_block state keys =
   simon_48_72_encrypt_iterate state keys"


definition simon_48_72_decrypt_block ::
  "24 word × 24 word ⇒ 24 word list ⇒ 24 word × 24 word" where
"simon_48_72_decrypt_block state keys =
   simon_48_72_decrypt_iterate state keys"


definition simon_48_72_encrypt ::
  "48 word ⇒ 24 word list ⇒ 48 word" where
"simon_48_72_encrypt plaintext keys =
  (let left  = ucast (drop_bit 24 plaintext);
       right = ucast plaintext;
       (c_l, c_r) = simon_48_72_encrypt_block (left, right) keys
   in or (push_bit 24 (ucast c_l)) (ucast c_r))"


definition simon_48_72_decrypt ::
  "48 word ⇒ 24 word list ⇒ 48 word" where
"simon_48_72_decrypt ciphertext keys =
  (let left  = ucast (drop_bit 24 ciphertext);
       right = ucast ciphertext;
       (p_l, p_r) = simon_48_72_decrypt_block (left, right) keys
   in or (push_bit 24 (ucast p_l)) (ucast p_r))"



end