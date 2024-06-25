module DRDoubleGreedy = struct
  type vector = float array

  let update_coordinate (x: vector) (v: int) (u: float) : vector =
    let new_x = Array.copy x in
    new_x.(v) <- u;
    new_x

  let find_u (f: vector -> float) (x: vector) (v: int) (delta: float) : float =
    let dummy_u = x.(v) +. 0.1 in
    dummy_u

  let dr_double_greedy (f: vector -> float) (a: vector) (b: vector) (delta: float) : vector =
    let n = Array.length a in
    let x = Array.copy a in
    let y = Array.copy b in
    
    for k = 1 to n do
      let v = k - 1 in  (* 0-based index for the coordinate being operated *)
      
      let ua = find_u (fun x' -> f (update_coordinate x' v x'.(v))) x v (delta /. float_of_int n) in
      let delta_a = f (update_coordinate x v ua) -. f x in
      
      let ub = find_u (fun y' -> f (update_coordinate y' v y'.(v))) y v (delta /. float_of_int n) in
      let delta_b = f (update_coordinate y v ub) -. f y in
      
      let weight_a = delta_a /. (delta_a +. delta_b) in
      let weight_b = delta_b /. (delta_a +. delta_b) in
      
      let new_value = weight_a *. ua +. weight_b *. ub in
      
      x.(v) <- new_value;
      y.(v) <- new_value;
    done;
    
    x

  let run (f: vector -> float) (a: vector) (b: vector) (delta: float) : vector =
    dr_double_greedy f a b delta
end