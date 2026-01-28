theory Simon_96_144
  imports "HOL-Library.Word" "HOL.Bit_Operations"
begin


definition simon_96_144_z3 :: int where 
  "simon_96_144_z3 = 0b11110000101100111001010001001000000111101001100011010111011011"


definition simon_96_144_F_function :: "48 word ⇒ 48 word" where
  "simon_96_144_F_function x = xor (and (word_rotl 1 x) (word_rotl 8 x)) (word_rotl 2 x)"


definition simon_96_144_encrypt_round :: "48 word ⇒ 48 word × 48 word ⇒ 48 word × 48 word" where
  "simon_96_144_encrypt_round k xy = (let (x, y) = xy in (xor (xor k (simon_96_144_F_function x)) y, x))"


definition simon_96_144_decrypt_round_inverse :: "48 word ⇒ 48 word × 48 word ⇒ 48 word × 48 word" where
  "simon_96_144_decrypt_round_inverse k xy_new = (let (x_new, y_new) = xy_new in (y_new, xor (xor x_new k) (simon_96_144_F_function y_new)))"


definition simon_96_144_rho_const :: "48 word" where
  "simon_96_144_rho_const = 0xFFFFFFFFFFFC"


function simon_96_144_gen_key_schedule_rec :: "48 word list ⇒ nat ⇒ 48 word list" where
  "simon_96_144_gen_key_schedule_rec ks i = (if i ≥ 54 then ks else
    let z_bit = bit simon_96_144_z3 (i - 3);
        rs_3 = word_rotr 3 (ks ! (i-1)); rs_1 = word_rotr 1 rs_3;
        new_k = xor (xor (xor (ks ! (i-3)) rs_3) rs_1) (xor (if z_bit then 1 else 0) simon_96_144_rho_const)
    in simon_96_144_gen_key_schedule_rec (ks @ [new_k]) (i+1))"
  by pat_completeness auto
termination by (relation "measure (λ(ks, i). 54 - i)") auto


definition simon_96_144_generate_key_schedule :: "48 word list ⇒ 48 word list" where
  "simon_96_144_generate_key_schedule init = simon_96_144_gen_key_schedule_rec init (length init)"


fun simon_96_144_encrypt_iterate :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "simon_96_144_encrypt_iterate st [] = st" 
| "simon_96_144_encrypt_iterate st (k#ks) = simon_96_144_encrypt_iterate (simon_96_144_encrypt_round k st) ks"


fun simon_96_144_decrypt_iterate :: "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
  "simon_96_144_decrypt_iterate st ks = foldl (λst_new k. simon_96_144_decrypt_round_inverse k st_new) st (rev ks)"


definition simon_96_144_encrypt_block ::
  "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
"simon_96_144_encrypt_block state keys =
   simon_96_144_encrypt_iterate state keys"


definition simon_96_144_decrypt_block ::
  "48 word × 48 word ⇒ 48 word list ⇒ 48 word × 48 word" where
"simon_96_144_decrypt_block state keys =
   simon_96_144_decrypt_iterate state keys"


definition simon_96_144_encrypt ::
  "96 word ⇒ 48 word list ⇒ 96 word" where
"simon_96_144_encrypt plaintext keys =
  (let left  = ucast (drop_bit 48 plaintext);
       right = ucast plaintext;
       (c_l, c_r) = simon_96_144_encrypt_block (left, right) keys
   in or (push_bit 48 (ucast c_l)) (ucast c_r))"


definition simon_96_144_decrypt ::
  "96 word ⇒ 48 word list ⇒ 96 word" where
"simon_96_144_decrypt ciphertext keys =
  (let left  = ucast (drop_bit 48 ciphertext);
       right = ucast ciphertext;
       (p_l, p_r) = simon_96_144_decrypt_block (left, right) keys
   in or (push_bit 48 (ucast p_l)) (ucast p_r))"


end