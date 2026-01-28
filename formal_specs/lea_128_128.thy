theory lea_128_128
  imports 
    "HOL-Library.Word" 
    "HOL.Bit_Operations"
begin

definition lea_128_128_word_size :: nat where "lea_128_128_word_size = 32"


definition lea_128_128_rounds    :: nat where "lea_128_128_rounds = 24"

definition lea_128_128_m :: nat where "lea_128_128_m = 4"


definition  lea_128_128_rol :: "32 word \<Rightarrow> nat \<Rightarrow> 32 word" where
  " lea_128_128_rol x r = word_rotl r x"


definition  lea_128_128_ror :: "32 word \<Rightarrow> nat \<Rightarrow> 32 word" where
  " lea_128_128_ror x r = word_rotr r x"



definition lea_128_128_delta :: "32 word list" where
  "lea_128_128_delta = 
    [ 0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 
      0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957 ]"


definition lea_128_128_delta_128 :: "32 word list" where
  "lea_128_128_delta_128 = take 4 lea_128_128_delta"


definition lea_128_128_extract_key_words :: "128 word \<Rightarrow> 32 word list" where
  "lea_128_128_extract_key_words k = 
    [ ucast k, 
      ucast (drop_bit 32 k), 
      ucast (drop_bit 64 k), 
      ucast (drop_bit 96 k) ]"


definition lea_128_128_step_key_expansion :: "32 word list \<Rightarrow> nat \<Rightarrow> (32 word list \<times> 32 word list)" where
  "lea_128_128_step_key_expansion t i = (
    let d = lea_128_128_delta_128 ! (i mod 4);
        t0 =  lea_128_128_rol (t!0 +  lea_128_128_rol d i) 1;
        t1 =  lea_128_128_rol (t!1 +  lea_128_128_rol d (i+1)) 3;
        t2 =  lea_128_128_rol (t!2 +  lea_128_128_rol d (i+2)) 6;
        t3 =  lea_128_128_rol (t!3 +  lea_128_128_rol d (i+3)) 11;
        rk = [t0, t1, t2, t1, t3, t1]
    in (rk, [t0, t1, t2, t3]))"


function lea_128_128_gen_round_keys_iter :: "32 word list \<Rightarrow> nat \<Rightarrow> 32 word list list \<Rightarrow> 32 word list list" where
  "lea_128_128_gen_round_keys_iter t i acc = (
    if i \<ge> lea_128_128_rounds then acc
    else
      let (rk, t_next) = lea_128_128_step_key_expansion t i
      in lea_128_128_gen_round_keys_iter t_next (i + 1) (acc @ [rk]))"
  by pat_completeness auto
termination by (relation "measure (\<lambda>(_, i, _). lea_128_128_rounds - i)") auto


definition lea_128_128_generate_round_keys :: "128 word \<Rightarrow> 32 word list list" where
  "lea_128_128_generate_round_keys key = 
    lea_128_128_gen_round_keys_iter (lea_128_128_extract_key_words key) 0 []"


definition lea_128_128_encrypt_round :: "32 word list \<Rightarrow> 32 word list \<Rightarrow> 32 word list" where
  "lea_128_128_encrypt_round state rk = (
    let x0 = state!0; x1 = state!1; x2 = state!2; x3 = state!3;
        y0 =  lea_128_128_rol ((xor x0 (rk!0)) + (xor x1 (rk!1))) 9;
        y1 =  lea_128_128_ror ((xor x1 (rk!2)) + (xor x2 (rk!3))) 5;
        y2 =  lea_128_128_ror ((xor x2 (rk!4)) + (xor x3 (rk!5))) 3;
        y3 = x0
    in [y0, y1, y2, y3])"



definition lea_128_128_decrypt_round :: "32 word list \<Rightarrow> 32 word list \<Rightarrow> 32 word list" where
  "lea_128_128_decrypt_round state rk = (
    let y0 = state!0; y1 = state!1; y2 = state!2; y3 = state!3;
        x0 = y3;
        x1 = xor ( lea_128_128_ror y0 9 - (xor x0 (rk!0))) (rk!1);
        x2 = xor ( lea_128_128_rol y1 5 - (xor x1 (rk!2))) (rk!3);
        x3 = xor ( lea_128_128_rol y2 3 - (xor x2 (rk!4))) (rk!5)
    in [x0, x1, x2, x3])"


fun lea_128_128_encrypt_iter :: "32 word list \<Rightarrow> 32 word list list \<Rightarrow> 32 word list" where
  "lea_128_128_encrypt_iter st [] = st"
| "lea_128_128_encrypt_iter st (rk#rks) = lea_128_128_encrypt_iter (lea_128_128_encrypt_round st rk) rks"


fun lea_128_128_decrypt_iter :: "32 word list \<Rightarrow> 32 word list list \<Rightarrow> 32 word list" where
  "lea_128_128_decrypt_iter st rks = foldl lea_128_128_decrypt_round st (rev rks)"


definition lea_128_128_encrypt_block :: 
  "128 word \<Rightarrow> 32 word list list \<Rightarrow> 128 word" where
  "lea_128_128_encrypt_block plaintext rks = (
    let state = [ucast plaintext, 
                 ucast (drop_bit 32 plaintext), 
                 ucast (drop_bit 64 plaintext), 
                 ucast (drop_bit 96 plaintext)];
        res = lea_128_128_encrypt_iter state rks
    in or (push_bit 96 (ucast (res!3))) 
          (or (push_bit 64 (ucast (res!2))) 
              (or (push_bit 32 (ucast (res!1))) 
                  (ucast (res!0)))))"


definition lea_128_128_decrypt_block :: 
  "128 word \<Rightarrow> 32 word list list \<Rightarrow> 128 word" where
  "lea_128_128_decrypt_block ciphertext rks = (
    let state = [ucast ciphertext, 
                 ucast (drop_bit 32 ciphertext), 
                 ucast (drop_bit 64 ciphertext), 
                 ucast (drop_bit 96 ciphertext)];
        res = lea_128_128_decrypt_iter state rks
    in or (push_bit 96 (ucast (res!3))) 
          (or (push_bit 64 (ucast (res!2))) 
              (or (push_bit 32 (ucast (res!1))) 
                  (ucast (res!0)))))"


definition lea_128_128_encrypt :: 
  "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
  "lea_128_128_encrypt plaintext key = 
    lea_128_128_encrypt_block plaintext (lea_128_128_generate_round_keys key)"


definition lea_128_128_decrypt :: 
  "128 word \<Rightarrow> 128 word \<Rightarrow> 128 word" where
  "lea_128_128_decrypt ciphertext key = 
    lea_128_128_decrypt_block ciphertext (lea_128_128_generate_round_keys key)"


end