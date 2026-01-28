theory gift_64_128
  imports
    "HOL-Library.Word"
    "HOL.Bit_Operations"
begin


definition gift_64_128_block_size :: nat where " gift_64_128_block_size = 64"
definition gift_64_128_key_size   :: nat where " gift_64_128_key_size = 128"
definition gift_64_128_rounds     :: nat where " gift_64_128_rounds = 28"


definition gift_64_128_sbox_table :: "nat list" where
  "gift_64_128_sbox_table =
   [1,10,4,12,6,15,3,9,2,13,11,7,5,0,8,14]"

definition gift_64_128_sbox_inv_table :: "nat list" where
  "gift_64_128_sbox_inv_table =
   [13,0,8,6,2,12,4,11,14,7,1,10,3,9,15,5]"

definition gift_64_128_sbox :: "4 word \<Rightarrow> 4 word" where
  "gift_64_128_sbox x = of_nat (gift_64_128_sbox_table ! unat x)"

definition gift_64_128_sbox_inv :: "4 word \<Rightarrow> 4 word" where
  "gift_64_128_sbox_inv x = of_nat (gift_64_128_sbox_inv_table ! unat x)"



definition gift_64_128_get_nibble :: "64 word \<Rightarrow> nat \<Rightarrow> 4 word" where
  "gift_64_128_get_nibble s i =
     ucast (take_bit 4 (drop_bit (4*i) s))"

definition gift_64_128_set_nibble :: "64 word \<Rightarrow> nat \<Rightarrow> 4 word \<Rightarrow> 64 word" where
  "gift_64_128_set_nibble s i v =
     or (and s (not (push_bit (4*i) (mask 4)))) (push_bit (4*i) (ucast v))"



definition gift_64_128_sbox_layer :: "64 word \<Rightarrow> 64 word" where
  "gift_64_128_sbox_layer s =
     fold (\<lambda>i acc. 
            gift_64_128_set_nibble acc i (gift_64_128_sbox (gift_64_128_get_nibble acc i)))
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
          s"

definition gift_64_128_sbox_layer_inv :: "64 word \<Rightarrow> 64 word" where
  "gift_64_128_sbox_layer_inv s =
     fold (\<lambda>i acc. 
            gift_64_128_set_nibble acc i (gift_64_128_sbox_inv (gift_64_128_get_nibble acc i)))
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
          s"



definition gift_64_128_perm_order :: "nat list" where
  "gift_64_128_perm_order =
   [0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
    4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
    8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]"


definition gift_64_128_perm_inv_order :: "nat list" where
  "gift_64_128_perm_inv_order =
   [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,
    1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,
    2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,
    3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]"

definition gift_64_128_get_bit :: "64 word \<Rightarrow> nat \<Rightarrow> 1 word" where
  "gift_64_128_get_bit s i = ucast (take_bit 1 (drop_bit i s))"

definition gift_64_128_set_bit :: "64 word \<Rightarrow> nat \<Rightarrow> 1 word \<Rightarrow> 64 word" where
  "gift_64_128_set_bit s i v =
     or (and s (not (push_bit i 1))) (push_bit i (ucast v))"

definition gift_64_128_perm_layer :: "64 word \<Rightarrow> 64 word" where
  "gift_64_128_perm_layer s =
     fold (\<lambda>i acc.
            if gift_64_128_get_bit s i = 1
            then gift_64_128_set_bit acc (gift_64_128_perm_order ! i) 1
            else acc)
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
           16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
           32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
           48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
          (0 :: 64 word)"

definition gift_64_128_perm_layer_inv :: "64 word \<Rightarrow> 64 word" where
  "gift_64_128_perm_layer_inv s =
     fold (\<lambda>i acc.
            if gift_64_128_get_bit s i = 1
            then gift_64_128_set_bit acc (gift_64_128_perm_inv_order ! i) 1
            else acc)
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
           16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
           32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
           48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
          (0 :: 64 word)"



definition gift_64_128_round_constants :: "nat list" where
  "gift_64_128_round_constants =
   [1,3,7,15,31,62,61,59,55,47,30,60,57,51,
    39,14,29,58,53,43,22,44,24,48,33,2,5,11]"



type_synonym key_state = "16 word list"
(* [W7 W6 W5 W4 W3 W2 W1 W0] *)

definition gift_64_128_key_setup :: "128 word \<Rightarrow> key_state" where
  "gift_64_128_key_setup k =
   [ ucast (take_bit 16 (drop_bit 112 k)),
     ucast (take_bit 16 (drop_bit 96 k)),
     ucast (take_bit 16 (drop_bit 80 k)),
     ucast (take_bit 16 (drop_bit 64 k)),
     ucast (take_bit 16 (drop_bit 48 k)),
     ucast (take_bit 16 (drop_bit 32 k)),
     ucast (take_bit 16 (drop_bit 16 k)),
     ucast (take_bit 16 k) ]"

definition gift_64_128_ror16 :: "16 word \<Rightarrow> nat \<Rightarrow> 16 word" where
  "gift_64_128_ror16 w n =
     or (drop_bit n w) (push_bit (16 - n) (take_bit n w))"

definition gift_64_128_key_update :: "key_state \<Rightarrow> key_state" where
  "gift_64_128_key_update ks =
   (case ks of
     [W7,W6,W5,W4,W3,W2,W1,W0] \<Rightarrow>
       [ gift_64_128_ror16 W1 12,   
         gift_64_128_ror16 W0 2,    
         W7,                 
         W6,                 
         W5,                 
         W4,                 
         W3,                 
         W2 ]                
   | _ \<Rightarrow> ks)"



definition gift_64_128_add_round_key_iter :: "64 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> 64 word" where
  "gift_64_128_add_round_key_iter s ks r =
     fold (\<lambda>i acc.
            xor (xor acc 
                  (push_bit (4*i)
                    (ucast (take_bit 1 (drop_bit i (ks ! 1))))))
                 (push_bit (4*i+1)
                    (ucast (take_bit 1 (drop_bit i (ks ! 5))))))
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
          s"


definition gift_64_128_add_round_constant :: "64 word \<Rightarrow> nat \<Rightarrow> 64 word" where
  "gift_64_128_add_round_constant s r =
     xor 
       (fold 
         (\<lambda>i acc. xor acc
           (push_bit (4*i+3)
             (ucast (take_bit 1 (drop_bit i
               (of_nat (gift_64_128_round_constants ! r) :: 6 word))))))
         [0,1,2,3,4,5]
         s)
       (push_bit 63 1)"


definition gift_64_128_add_round_key :: "64 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> 64 word" where
  "gift_64_128_add_round_key s ks r =
     gift_64_128_add_round_constant
       (gift_64_128_add_round_key_iter s ks r) r"



definition gift_64_128_encrypt_round ::
  "64 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> (64 word \<times> key_state)" where
  "gift_64_128_encrypt_round s ks r =
   (let s1 = gift_64_128_sbox_layer s;
        s2 = gift_64_128_perm_layer s1;
        s3 = gift_64_128_add_round_key s2 ks r;
        ks' = gift_64_128_key_update ks
    in (s3, ks'))"

definition gift_64_128_decrypt_round ::
  "64 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> (64 word \<times> key_state)" where
  "gift_64_128_decrypt_round s ks r =
   (let s1 = gift_64_128_add_round_key s ks r;   
        s2 = gift_64_128_perm_layer_inv s1;      
        s3 = gift_64_128_sbox_layer_inv s2       
    in (s3, ks))"  (* Note: for decryption, we use the same key, not updated *)



fun gift_64_128_generate_round_keys :: "key_state \<Rightarrow> nat \<Rightarrow> key_state list" where
  "gift_64_128_generate_round_keys ks 0 = []"
| "gift_64_128_generate_round_keys ks (Suc n) = 
     ks # gift_64_128_generate_round_keys (gift_64_128_key_update ks) n"


definition gift_64_128_key_schedule :: "128 word \<Rightarrow> key_state list" where
  "gift_64_128_key_schedule k = 
     gift_64_128_generate_round_keys (gift_64_128_key_setup k)  gift_64_128_rounds"



fun gift_64_128_encrypt_iter ::
  "64 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> 64 word" where
  "gift_64_128_encrypt_iter s ks r 0 = s"
| "gift_64_128_encrypt_iter s ks r (Suc n) =
     (let (s', ks') = gift_64_128_encrypt_round s ks r
      in gift_64_128_encrypt_iter s' ks' (r + 1) n)"



fun gift_64_128_decrypt_iter ::
  "64 word \<Rightarrow> key_state list \<Rightarrow> nat \<Rightarrow> 64 word" where
  "gift_64_128_decrypt_iter s [] _ = s"
| "gift_64_128_decrypt_iter s (k#ks) r =
     (if r = 0 then gift_64_128_add_round_key s k r
      else let (s', _) = gift_64_128_decrypt_round s k r
           in gift_64_128_decrypt_iter s' ks (r-1))"


definition gift_64_128_encrypt_iterate ::
  "64 word \<Rightarrow> key_state \<Rightarrow> 64 word" where
"gift_64_128_encrypt_iterate s ks =
   gift_64_128_encrypt_iter s ks 0 gift_64_128_rounds"


definition gift_64_128_decrypt_iterate ::
  "64 word \<Rightarrow> key_state list \<Rightarrow> 64 word" where
"gift_64_128_decrypt_iterate s keys =
   gift_64_128_decrypt_iter s (rev keys) (gift_64_128_rounds - 1)"




definition gift_64_128_encrypt_block :: "64 word \<Rightarrow> 128 word \<Rightarrow> 64 word" where
"gift_64_128_encrypt_block plaintext key =
   gift_64_128_encrypt_iterate plaintext (gift_64_128_key_setup key)"


definition gift_64_128_decrypt_block :: "64 word \<Rightarrow> 128 word \<Rightarrow> 64 word" where
  "gift_64_128_decrypt_block ciphertext key = 
      gift_64_128_decrypt_iterate ciphertext (gift_64_128_key_schedule key) "



definition gift_64_128_encrypt ::
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
"gift_64_128_encrypt key plaintext =
   gift_64_128_encrypt_block plaintext key"


definition gift_64_128_decrypt ::
  "128 word \<Rightarrow> 64 word \<Rightarrow> 64 word" where
"gift_64_128_decrypt key ciphertext =
   gift_64_128_decrypt_block ciphertext key"



definition test_key :: "128 word" where
  "test_key = 0x000102030405060708090A0B0C0D0E0F"

definition test_plaintext :: "64 word" where
  "test_plaintext = 0x0001020304050607"

definition expected_ciphertext :: "64 word" where
  "expected_ciphertext = 0x5D98C3A9C5F50406"



end