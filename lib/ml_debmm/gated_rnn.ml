open Torch

let hidden_size = 1024
let input_size = 128
let dynamic_size = (input_size - 1) / 2

let model () =
  let state = Tensor.zeros [hidden_size] in
  
  let memory = Layer.Linear.create ~in_dim:input_size ~out_dim:dynamic_size in
  
  let gate = Layer.Sequential.of_list [
    Layer.Linear.create ~in_dim:(input_size + hidden_size) ~out_dim:hidden_size;
    Layer.Sigmoid.create ()
  ] in
  
  let static = Layer.Linear.create ~in_dim:(input_size + hidden_size) ~out_dim:(hidden_size - dynamic_size) in
  
  let energy = Layer.Linear.create ~in_dim:hidden_size ~out_dim:1 in
  
  let forward x =
    let rec loop state hop =
      if hop >= 5 then state
      else
        let z = Tensor.cat [x; state] ~dim:0 in
        let dynamic_part = Layer.Linear.forward memory x in
        let static_part = Layer.Linear.forward static z in
        let c = Tensor.tanh (Tensor.cat [dynamic_part; static_part] ~dim:0) in
        let u = Layer.Sequential.forward gate z in
        let new_state = Tensor.(u * c + ((Scalar.f 1. - u) * state)) in
        loop new_state (hop + 1)
    in
    let final_state = loop state 0 in
    Layer.Linear.forward energy final_state
  in
  
  forward