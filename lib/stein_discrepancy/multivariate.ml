open Lwt
open Domainslib

module SpannerSteinDiscrepancy = struct
  type coordinate_bounds = (float * float) array
  type graph = (int * int) list

  let compute_sparse_2_spanner support =
    List.init (List.length support - 1) (fun i -> (i, i + 1))

  let solve_coordinate_linear_program j g2 (alpha, beta) =
    Random.float (beta -. alpha) +. alpha

  let compute_discrepancy q bounds =
    let d = Array.length bounds in
    let g2 = compute_sparse_2_spanner q in
    
    let pool = Task.setup_pool ~num_domains:(Domain.recommended_domain_count ()) () in
    
    let results = 
      Array.init d (fun j ->
        Task.async pool (fun _ ->
          solve_coordinate_linear_program j g2 bounds.(j)
        )
      )
    in
    
    let sum = 
      Array.fold_left (fun acc task ->
        acc +. Task.await pool task
      ) 0.0 results
    in
    
    Task.teardown_pool pool;
    sum

end