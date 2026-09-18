/* ---------------------------------------------------------------------------
 * tachys — variational benchmark database
 * ---------------------------------------------------------------------------
 * Every entry of BENCHMARKS is one table: a (model, lattice, size, point)
 * combination. The page builds its Model / Lattice / Size / Point selectors
 * directly from this list, so adding a new system is purely a data edit --
 * append an object below and it shows up in the UI.
 *
 * Table schema
 * ------------
 *   model     string   e.g. "J1-J2 Heisenberg"        (selector 1)
 *   lattice   string   e.g. "Square"                  (selector 2)
 *   size      string   e.g. "10 x 10"                 (selector 3)
 *   point     string   coupling / parameter point     (selector 4)
 *   sites     number   number of sites N
 *   boundary  string   boundary conditions
 *   quantity  string   what the "Energy" column reports
 *   hamiltonian string LaTeX-free inline formula shown under the title
 *   note      string   optional caption below the table
 *
 * Row schema
 * ----------
 *   energy    number   value used for sorting and for the ranking
 *   display   string   energy exactly as printed in the source table
 *   variance  string   energy variance as printed, e.g. "0.0021(3)".
 *                      Optional -- omit it and the cell shows an em dash.
 *   wf        string   wave function / method name
 *   kind      string   one of "NQS" | "Hybrid" | "Tensor network" | "Variational"
 *   params    string   number of variational parameters ("" => not available)
 *   prior     string   "Yes" | "No" | "" -- kept for the CSV/archive, not shown
 *   ref       string   short citation
 *   url       string   optional DOI / proceedings link ("" => plain text)
 *   year      number
 *   tag       string   optional badge, e.g. "Reference paper"
 * ------------------------------------------------------------------------- */

const BENCHMARKS = [
  {
    model: "J1-J2 Heisenberg",
    lattice: "Square",
    size: "10 x 10",
    point: "J2/J1 = 0.5",
    sites: 100,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    prior_label: "Marshall prior",
    note:
      "Maximally frustrated point of the spin-1/2 J1-J2 Heisenberg model — the standard " +
      "hard benchmark for variational wave functions in two dimensions. Values are " +
      "reproduced from Table 1 of Rende et al., Commun. Phys. 7, 260 (2024).",
    source: {
      cite: "R. Rende, L. L. Viteritti, L. Bardone, F. Becca, S. Goldt, Commun. Phys. 7, 260 (2024)",
      url: "https://doi.org/10.1038/s42005-024-01732-4",
    },
    rows: [
      {
        energy: -0.48941, display: "−0.48941(1)", wf: "MLP", kind: "NQS",
        params: "893994", prior: "",
        ref: "Ledinauskas & Anisimovas, SciPost Phys. 15, 229",
        url: "https://doi.org/10.21468/SciPostPhys.15.6.229", year: 2023,
      },
      {
        energy: -0.494757, display: "−0.494757(12)", wf: "CNN", kind: "NQS",
        params: "", prior: "No",
        ref: "Szabó & Castelnovo, Phys. Rev. Res. 2, 033075",
        url: "https://doi.org/10.1103/PhysRevResearch.2.033075", year: 2020,
      },
      {
        energy: -0.4947359, display: "−0.4947359(1)", wf: "Shallow CNN", kind: "NQS",
        params: "11009", prior: "",
        ref: "Liang et al., Phys. Rev. B 98, 104426",
        url: "https://doi.org/10.1103/PhysRevB.98.104426", year: 2018,
      },
      {
        energy: -0.49516, display: "−0.49516(1)", wf: "Deep CNN", kind: "NQS",
        params: "7676", prior: "Yes",
        ref: "Choo, Neupert & Carleo, Phys. Rev. B 100, 125124",
        url: "https://doi.org/10.1103/PhysRevB.100.125124", year: 2019,
      },
      {
        energy: -0.495502, display: "−0.495502(1)", wf: "PEPS + Deep CNN", kind: "Hybrid",
        params: "3531", prior: "No",
        ref: "Liang, Dong & He, Phys. Rev. B 103, 035138",
        url: "https://doi.org/10.1103/PhysRevB.103.035138", year: 2021,
      },
      {
        energy: -0.495530, display: "−0.495530", wf: "DMRG", kind: "Tensor network",
        params: "8192 SU(2) states", prior: "No",
        ref: "Gong et al., Phys. Rev. Lett. 113, 027201",
        url: "https://doi.org/10.1103/PhysRevLett.113.027201", year: 2014,
      },
      {
        energy: -0.495627, display: "−0.495627(6)", wf: "aCNN", kind: "NQS",
        params: "6538", prior: "Yes",
        ref: "Wang, He & Lu, Phys. Rev. B 109, 245120",
        url: "https://doi.org/10.1103/PhysRevB.109.245120", year: 2023,
      },
      {
        energy: -0.49575, display: "−0.49575(3)", wf: "RBM-fermionic", kind: "Hybrid",
        params: "2000", prior: "Yes",
        ref: "Ferrari, Becca & Carrasquilla, Phys. Rev. B 100, 125131",
        url: "https://doi.org/10.1103/PhysRevB.100.125131", year: 2019,
      },
      {
        energy: -0.49586, display: "−0.49586(4)", wf: "CNN", kind: "NQS",
        params: "10952", prior: "Yes",
        ref: "Reh, Schmitt & Gärttner, Phys. Rev. B 107, 195115",
        url: "https://doi.org/10.1103/PhysRevB.107.195115", year: 2023,
      },
      {
        energy: -0.4968, display: "−0.4968(4)", wf: "RBM (p = 1)", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Chen, Hendry, Weinberg & Feiguin, NeurIPS 35, 7490",
        url: "https://proceedings.neurips.cc/paper_files/paper/2022/file/3173c427cb4ed2d5eaab029c17f221ae-Paper-Conference.pdf",
        year: 2022,
      },
      {
        energy: -0.49717, display: "−0.49717(1)", wf: "Deep CNN", kind: "NQS",
        params: "106529", prior: "Yes",
        ref: "Li et al., IEEE Trans. Parallel Distrib. Syst. 33, 2846",
        url: "", year: 2022,
      },
      {
        energy: -0.497437, display: "−0.497437(7)", wf: "GCNN", kind: "NQS",
        params: "67548", prior: "No",
        ref: "Roth, Szabó & MacDonald, Phys. Rev. B 108, 054410",
        url: "https://doi.org/10.1103/PhysRevB.108.054410", year: 2023,
      },
      {
        energy: -0.497468, display: "−0.497468(1)", wf: "Deep CNN", kind: "NQS",
        params: "421953", prior: "Yes",
        ref: "Liang et al., Mach. Learn.: Sci. Technol. 4, 015035",
        url: "", year: 2022,
      },
      {
        energy: -0.4975490, display: "−0.4975490(2)", wf: "VMC (p = 2)", kind: "Variational",
        params: "5", prior: "Yes",
        ref: "Hu, Becca, Parola & Sorella, Phys. Rev. B 88, 060402",
        url: "https://doi.org/10.1103/PhysRevB.88.060402", year: 2013,
      },
      {
        energy: -0.497627, display: "−0.497627(1)", wf: "Deep CNN", kind: "NQS",
        params: "146320", prior: "Yes",
        ref: "Chen & Heyl, Nat. Phys. 20, 1476",
        url: "https://doi.org/10.1038/s41567-024-02566-1", year: 2023,
      },
      {
        energy: -0.497629, display: "−0.497629(1)", wf: "RBM + PP", kind: "Hybrid",
        params: "13132", prior: "Yes",
        ref: "Nomura & Imada, Phys. Rev. X 11, 031034",
        url: "https://doi.org/10.1103/PhysRevX.11.031034", year: 2021,
      },
      {
        energy: -0.497634, display: "−0.497634(1)", wf: "Deep ViT", kind: "NQS",
        params: "267720", prior: "No",
        ref: "Rende et al., Commun. Phys. 7, 260",
        url: "https://doi.org/10.1038/s42005-024-01732-4", year: 2023,
        tag: "Reference paper",
      },
    ],
  },
];
