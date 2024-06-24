open Torch

module NaiveMeanField = struct
  type t = {
    n: int;
    f: Tensor.t -> Tensor.t;
    k: int;
    t: float;
  }

  let create n f k t =
    { n; f; k; t }

  let calculate_variational_index game x0 =
    let x = ref x0 in
    for epoch = 1 to game.k do
      for i = 1 to game.n do
        let v_i = Tensor.slice ~dim:0 ~start:i ~end_:(i+1) !x in
        let grad = Tensor.grad_of_fn game.f v_i in
        let updated_v_i = Tensor.sigmoid (Tensor.div_scalar grad game.t) in
        x := Tensor.copy_ ~src:updated_v_i ~dst:(Tensor.slice ~dim:0 ~start:i ~end_:(i+1) !x)
      done
    done;
    !x

  let variational_index game x0 =
    let x_star = calculate_variational_index game x0 in
    Tensor.log (Tensor.div_scalar x_star (Tensor.sub_scalar (Tensor.ones_like x_star) x_star))
end
