theory gift_128_128
  imports
    "HOL-Library.Word"
    "HOL.Bit_Operations"
begin


definition gift_128_128_block_size :: nat where "gift_128_128_block_size = 128"
definition gift_128_128_key_size   :: nat where "gift_128_128_key_size = 128"
definition gift_128_128_rounds     :: nat where "gift_128_128_rounds = 40"


definition gift_128_128_sbox_table :: "nat list" where
  "gift_128_128_sbox_table =
   [1,10,4,12,6,15,3,9,2,13,11,7,5,0,8,14]"


definition gift_128_128_sbox_inv_table :: "nat list" where
  "gift_128_128_sbox_inv_table =
   [13,0,8,6,2,12,4,11,14,7,1,10,3,9,15,5]"


definition gift_128_128_sbox :: "4 word \<Rightarrow> 4 word" where
  "gift_128_128_sbox x = of_nat (gift_128_128_sbox_table ! unat x)"


definition gift_128_128_sbox_inv :: "4 word \<Rightarrow> 4 word" where
  "gift_128_128_sbox_inv x = of_nat (gift_128_128_sbox_inv_table ! unat x)"


definition gift_128_128_get_nibble :: "128 word \<Rightarrow> nat \<Rightarrow> 4 word" where
  "gift_128_128_get_nibble s i =
     ucast (take_bit 4 (drop_bit (4*i) s))"


definition gift_128_128_set_nibble :: "128 word \<Rightarrow> nat \<Rightarrow> 4 word \<Rightarrow> 128 word" where
  "gift_128_128_set_nibble s i v =
     or (and s (not (push_bit (4*i) (mask 4)))) (push_bit (4*i) (ucast v))"


definition gift_128_128_sbox_layer :: "128 word \<Rightarrow> 128 word" where
  "gift_128_128_sbox_layer s =
     fold (\<lambda>i acc. 
            gift_128_128_set_nibble acc i (gift_128_128_sbox (gift_128_128_get_nibble acc i)))
          [0..<32]
          s"


definition gift_128_128_sbox_layer_inv :: "128 word \<Rightarrow> 128 word" where
  "gift_128_128_sbox_layer_inv s =
     fold (\<lambda>i acc. 
            gift_128_128_set_nibble acc i (gift_128_128_sbox_inv (gift_128_128_get_nibble acc i)))
          [0..<32]
          s"

definition gift_128_128_perm_order :: "nat list" where
  "gift_128_128_perm_order = map (\<lambda>i. if i = 127 then 127 else (32 * i) mod 127) [0..<128]"


definition gift_128_128_perm_inv_order :: "nat list" where
  "gift_128_128_perm_inv_order = map (\<lambda>i. if i = 127 then 127 else (4 * i) mod 127) [0..<128]"


definition gift_128_128_get_bit :: "128 word \<Rightarrow> nat \<Rightarrow> 1 word" where
  "gift_128_128_get_bit s i = ucast (take_bit 1 (drop_bit i s))"


definition gift_128_128_set_bit :: "128 word \<Rightarrow> nat \<Rightarrow> 1 word \<Rightarrow> 128 word" where
  "gift_128_128_set_bit s i v =
     or (and s (not (push_bit i 1))) (push_bit i (ucast v))"


definition gift_128_128_perm_layer :: "128 word \<Rightarrow> 128 word" where
  "gift_128_128_perm_layer s =
     fold (\<lambda>i acc.
            if gift_128_128_get_bit s i = 1
            then gift_128_128_set_bit acc (gift_128_128_perm_order ! i) 1
            else acc)
          [0..<128]
          (0 :: 128 word)"

definition gift_128_128_perm_layer_inv :: "128 word \<Rightarrow> 128 word" where
  "gift_128_128_perm_layer_inv s =
     fold (\<lambda>i acc.
            if gift_128_128_get_bit s i = 1
            then gift_128_128_set_bit acc (gift_128_128_perm_inv_order ! i) 1
            else acc)
          [0..<128]
          (0 :: 128 word)"

definition gift_128_128_round_constants :: "nat list" where
  "gift_128_128_round_constants =
   [1,3,7,15,31,62,61,59,55,47,30,60,57,51,39,14,29,58,53,43,
    22,44,24,48,33,2,5,11,23,46,28,56,49,35,6,13,27,54,45,26]"

type_synonym key_state = "16 word list"

definition gift_128_128_key_setup :: "128 word \<Rightarrow> key_state" where
  "gift_128_128_key_setup k =
   [ ucast (take_bit 16 (drop_bit 112 k)),
     ucast (take_bit 16 (drop_bit 96 k)),
     ucast (take_bit 16 (drop_bit 80 k)),
     ucast (take_bit 16 (drop_bit 64 k)),
     ucast (take_bit 16 (drop_bit 48 k)),
     ucast (take_bit 16 (drop_bit 32 k)),
     ucast (take_bit 16 (drop_bit 16 k)),
     ucast (take_bit 16 k) ]"


definition gift_128_128_ror16 :: "16 word \<Rightarrow> nat \<Rightarrow> 16 word" where
  "gift_128_128_ror16 w n =
     or (drop_bit n w) (push_bit (16 - n) (take_bit n w))"


definition gift_128_128_key_update :: "key_state \<Rightarrow> key_state" where
  "gift_128_128_key_update ks =
   (case ks of
     [W7,W6,W5,W4,W3,W2,W1,W0] \<Rightarrow>
       [ gift_128_128_ror16 W1 12,   
         gift_128_128_ror16 W0 2,    
         W7, W6, W5, W4, W3, W2 ]
   | _ \<Rightarrow> ks)"

definition gift_128_128_add_round_key_iter :: "128 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> 128 word" where
  "gift_128_128_add_round_key_iter s ks r =
     fold (\<lambda>i acc.
            xor (xor acc 
                  (push_bit (4*i)
                    (ucast (take_bit 1 (drop_bit i (ks ! 1))))))
                  (push_bit (4*i+1)
                    (ucast (take_bit 1 (drop_bit i (ks ! 5))))))
          [0..<32]
          s"


definition gift_128_128_add_round_constant :: "128 word \<Rightarrow> nat \<Rightarrow> 128 word" where
  "gift_128_128_add_round_constant s r =
     xor 
       (fold 
         (\<lambda>i acc. xor acc
           (push_bit (4*i+3)
             (ucast (take_bit 1 (drop_bit i
               (of_nat (gift_128_128_round_constants ! r) :: 6 word))))))
         [0..<6]
         s)
       (push_bit 127 1)"


definition gift_128_128_add_round_key :: "128 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> 128 word" where
  "gift_128_128_add_round_key s ks r =
     gift_128_128_add_round_constant
       (gift_128_128_add_round_key_iter s ks r) r"

definition gift_128_128_encrypt_round ::
  "128 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> (128 word \<times> key_state)" where
  "gift_128_128_encrypt_round s ks r =
   (let s1 = gift_128_128_sbox_layer s;
        s2 = gift_128_128_perm_layer s1;
        s3 = gift_128_128_add_round_key s2 ks r;
        ks' = gift_128_128_key_update ks
    in (s3, ks'))"


definition gift_128_128_decrypt_round ::
  "128 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> (128 word \<times> key_state)" where
  "gift_128_128_decrypt_round s ks r =
   (let s1 = gift_128_128_add_round_key s ks r;   
        s2 = gift_128_128_perm_layer_inv s1;      
        s3 = gift_128_128_sbox_layer_inv s2       
    in (s3, ks))"


fun gift_128_128_generate_round_keys :: "key_state \<Rightarrow> nat \<Rightarrow> key_state list" where
  "gift_128_128_generate_round_keys ks 0 = []"
| "gift_128_128_generate_round_keys ks (Suc n) = 
     ks # gift_128_128_generate_round_keys (gift_128_128_key_update ks) n"


definition gift_128_128_key_schedule :: "128 word \<Rightarrow> key_state list" where
  "gift_128_128_key_schedule k = 
     gift_128_128_generate_round_keys (gift_128_128_key_setup k) 40"

fun gift_128_128_encrypt_iter ::
  "128 word \<Rightarrow> key_state \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> 128 word" where
  "gift_128_128_encrypt_iter s ks r 0 = s"
| "gift_128_128_encrypt_iter s ks r (Suc n) =
     (let (s', ks') = gift_128_128_encrypt_round s ks r
      in gift_128_128_encrypt_iter s' ks' (r + 1) n)"


fun gift_128_128_decrypt_iter ::
  "128 word \<Rightarrow> key_state list \<Rightarrow> nat \<Rightarrow> 128 word" where
  "gift_128_128_decrypt_iter s [] _ = s"
| "gift_128_128_decrypt_iter s (k#ks) r =
     (if r = 0 then gift_128_128_add_round_key s k r
      else let (s', _) = gift_128_128_decrypt_round s k r
            in gift_128_128_decrypt_iter s' ks (r-1))"


definition gift_128_128_encrypt_iterate ::
  "128 word \<Rightarrow> key_state \<Rightarrow> 128 word" where
"gift_128_128_encrypt_iterate s ks =
   gift_128_128_encrypt_iter s ks 0 40"


definition gift_128_128_decrypt_iterate ::
  "128 word \<Rightarrow> key_state list \<Rightarrow> 128 word" where
"gift_128_128_decrypt_iterate s keys =
   gift_128_128_decrypt_iter s (rev keys) (40 - 1)"


definition gift_128_128_encrypt_block :: "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
"gift_128_128_encrypt_block plaintext key =
   gift_128_128_encrypt_iterate plaintext (gift_128_128_key_setup key)"


definition gift_128_128_decrypt_block :: "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
  "gift_128_128_decrypt_block ciphertext key = 
      gift_128_128_decrypt_iterate ciphertext (gift_128_128_key_schedule key) "

definition gift_128_128_encrypt :: "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
"gift_128_128_encrypt key plaintext = gift_128_128_encrypt_block plaintext key"


definition gift_128_128_decrypt :: "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
"gift_128_128_decrypt key ciphertext = gift_128_128_decrypt_block ciphertext key"



end