open Torch

module NaiveNCELoss = struct
  type t = {
    k: int;
    q: Tensor.t -> Tensor.t -> Tensor.t;
    p_theta: Tensor.t -> Tensor.t -> Tensor.t;
  }

  let create k q p_theta =
    { k; q; p_theta }

  let estimate_loss nce_loss x =
    let n = Tensor.shape x |> List.hd in
    
    let initial_loss = 
      Tensor.arange n
      |> Tensor.to_type ~type_:(T Float)
      |> Tensor.map (fun t ->
        let x_t = Tensor.select x ~dim:0 ~index:(int_of_float t) in
        let x_not_t = Tensor.cat [Tensor.slice x ~dim:0 ~start:0 ~end_:(int_of_float t);
                                  Tensor.slice x ~dim:0 ~start:((int_of_float t) + 1) ~end_:n] ~dim:0 in
        let numerator = Tensor.mul_scalar (nce_loss.p_theta x_t x_not_t) (float_of_int n) in
        let denominator = Tensor.(add 
          (mul_scalar (nce_loss.p_theta x_t x_not_t) (float_of_int n))
          (mul_scalar (nce_loss.q x_t x_not_t) (float_of_int nce_loss.k))) in
        Tensor.neg (Tensor.log (Tensor.div numerator denominator))
      )
      |> Tensor.sum in
    
    let negative_samples = 
      Tensor.randint ~high:n ~size:[nce_loss.k] ~options:(T Int64, Cpu)
      |> Tensor.to_type ~type_:(T Float) in
    
    let negative_loss = 
      Tensor.map negative_samples (fun t ->
        let x_t = Tensor.select x ~dim:0 ~index:(int_of_float t) in
        let x_not_t = Tensor.cat [Tensor.slice x ~dim:0 ~start:0 ~end_:(int_of_float t);
                                  Tensor.slice x ~dim:0 ~start:((int_of_float t) + 1) ~end_:n] ~dim:0 in
        let numerator = Tensor.mul_scalar (nce_loss.q x_t x_not_t) (float_of_int nce_loss.k) in
        let denominator = Tensor.(add 
          (mul_scalar (nce_loss.p_theta x_t x_not_t) (float_of_int n))
          (mul_scalar (nce_loss.q x_t x_not_t) (float_of_int nce_loss.k))) in
        Tensor.neg (Tensor.log (Tensor.div numerator denominator))
      )
      |> Tensor.sum in
    
    Tensor.add initial_loss negative_loss
end