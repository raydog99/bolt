open Torch

module PC_Sampler = struct
  type t = {
    n: int;
    m: int;
    predictor: Tensor.t -> Tensor.t;
    corrector: Tensor.t -> Tensor.t;
    pt: Tensor.t -> Tensor.t;
  }

  let create n m predictor corrector pt =
    { n; m; predictor; corrector; pt }

  let sample t =
    let x = ref (t.pt (Tensor.zeros [])) in
    for i = t.n - 1 downto 0 do
      x := t.predictor !x;
      for _ = 1 to t.m do
        x := t.corrector !x
      done
    done;
    !x

end