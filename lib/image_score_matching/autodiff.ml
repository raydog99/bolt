open Torch

module Node = struct
  type t = {
    mutable g: Tensor.t;
    mutable delta: Tensor.t;
    mutable delta_prime: Tensor.t;
    mutable partial_k_delta: Tensor.t;
    mutable partial_l_delta: Tensor.t;
    mutable partial_l_delta_prime: Tensor.t;
    mutable partial_j_g: Tensor.t;
    parents: int list;
    children: int list;
  }

  let create parents children =
    {
      g = Tensor.zeros [];
      delta = Tensor.zeros [];
      delta_prime = Tensor.zeros [];
      partial_k_delta = Tensor.zeros [];
      partial_l_delta = Tensor.zeros [];
      partial_l_delta_prime = Tensor.zeros [];
      partial_j_g = Tensor.zeros [];
      parents;
      children;
    }
end

module Network = struct
  type t = {
    nodes: Node.t array;
    weights: Tensor.t;
    mutable k: Tensor.t;
    mutable l: Tensor.t;
  }

  let create node_configs weights =
    let nodes = Array.of_list (List.map (fun (parents, children) -> Node.create parents children) node_configs) in
    { nodes; weights; k = Tensor.zeros []; l = Tensor.zeros [] }

  let forward_propagation t x =
    for j = Array.length t.nodes - 1 downto 0 do
      let node = t.nodes.(j) in
      if j < Tensor.shape x |> List.hd then
        node.g <- Tensor.select x 0 j
      else
        node.g <- Tensor.zeros [];
        List.iter (fun i ->
          node.g <- Tensor.add node.g (Tensor.mm t.nodes.(i).g t.weights)
        ) node.parents
    done

  let backpropagation t =
    let n = Array.length t.nodes - 1 in
    t.nodes.(n).delta <- Tensor.ones [];
    t.nodes.(n).delta_prime <- Tensor.zeros [];
    for j = n - 1 downto 0 do
      let node = t.nodes.(j) in
      node.delta <- Tensor.zeros [];
      node.delta_prime <- Tensor.zeros [];
      List.iter (fun k ->
        let child = t.nodes.(k) in
        node.delta <- Tensor.add node.delta (Tensor.mm child.delta t.weights);
        node.delta_prime <- Tensor.add node.delta_prime
          (Tensor.add
             (Tensor.mm child.delta (Tensor.pow t.weights 2))
             (Tensor.mm child.delta_prime (Tensor.pow t.weights 2)))
      ) node.children
    done

  let compute_loss t =
    let d = Tensor.shape t.nodes.(0).g |> List.hd in
    t.k <- Tensor.zeros [];
    t.l <- Tensor.zeros [];
    for j = 0 to d - 1 do
      let node = t.nodes.(j) in
      t.k <- Tensor.add t.k (Tensor.pow node.delta 2);
      t.l <- Tensor.add t.l (Tensor.add node.delta_prime (Tensor.pow node.delta_prime 2))
    done;
    t.k <- Tensor.mul_scalar t.k (-0.5);
    t.l <- Tensor.neg t.l

  let sm_forward_propagation t =
    for j = 0 to Array.length t.nodes - 1 do
      let node = t.nodes.(j) in
      if j < Tensor.shape t.nodes.(0).g |> List.hd then
        node.partial_k_delta <- node.delta;
      else begin
        node.partial_k_delta <- Tensor.zeros [];
        node.partial_l_delta <- Tensor.zeros [];
        node.partial_l_delta_prime <- Tensor.zeros [];
        List.iter (fun i ->
          let parent = t.nodes.(i) in
          node.partial_k_delta <- Tensor.add node.partial_k_delta
            (Tensor.mm parent.partial_k_delta t.weights);
          node.partial_l_delta <- Tensor.add node.partial_l_delta
            (Tensor.add
               (Tensor.mm parent.partial_l_delta_prime (Tensor.pow t.weights 2))
               (Tensor.mm parent.partial_l_delta t.weights));
          node.partial_l_delta_prime <- Tensor.add node.partial_l_delta_prime
            (Tensor.mm parent.partial_l_delta_prime (Tensor.pow t.weights 2))
        ) node.parents
      end
    done

  let sm_backward_propagation t =
    for j = Array.length t.nodes - 1 downto Tensor.shape t.nodes.(0).g |> List.hd do
      let node = t.nodes.(j) in
      node.partial_j_g <- Tensor.zeros [];
      List.iter (fun k ->
        let child = t.nodes.(k) in
        node.partial_j_g <- Tensor.add node.partial_j_g
          (Tensor.add
             (Tensor.mm child.partial_j_g t.weights)
             (Tensor.add
                (Tensor.mm node.partial_k_delta (Tensor.pow t.weights 2))
                (Tensor.add
                   (Tensor.mm node.partial_l_delta (Tensor.pow t.weights 2))
                   (Tensor.add
                      (Tensor.mul_scalar
                         (Tensor.mm node.partial_l_delta_prime
                            (Tensor.mul t.weights (Tensor.pow t.weights 2)))
                         2.)
                      (Tensor.mm node.partial_l_delta_prime (Tensor.pow t.weights 3))))))
      ) node.children
    done

  let compute_weight_gradients t =
    let gradients = Tensor.zeros (Tensor.shape t.weights) in
    for j = Tensor.shape t.nodes.(0).g |> List.hd to Array.length t.nodes - 1 do
      let node = t.nodes.(j) in
      List.iter (fun i ->
        let parent = t.nodes.(i) in
        gradients <- Tensor.add gradients
          (Tensor.add
             (Tensor.mm node.partial_j_g parent.g)
             (Tensor.add
                (Tensor.mm node.partial_k_delta parent.delta)
                (Tensor.add
                   (Tensor.mm node.partial_l_delta parent.delta)
                   (Tensor.mm node.partial_l_delta_prime parent.delta_prime))))
      ) node.parents
    done;
    gradients
end

let score_matching_loss network x =
  Network.forward_propagation network x;
  Network.backpropagation network;
  Network.compute_loss network;
  Network.sm_forward_propagation network;
  Network.sm_backward_propagation network;
  Tensor.add network.k network.l

let score_matching_gradients network x =
  let _ = score_matching_loss network x in
  Network.compute_weight_gradients network