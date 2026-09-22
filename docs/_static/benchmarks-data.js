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
 *   hamiltonian string plain-text formula shown under the title -- also the
 *                      fallback when KaTeX is unavailable
 *   hamiltonianTex string the same formula as LaTeX, typeset with MathJax
 *   pointLabel string optional label for the "point" chip (default "Coupling")
 *   extra     array    optional [{k, v}] chips appended to the system card
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
 *   prior     string   "Yes" | "No" | "" -- kept as archive metadata, not shown
 *   ref       string   short citation
 *   url       string   optional DOI / proceedings link ("" => plain text)
 *   year      number
 *   tachys    bool     optional -- true marks a result reproduced with tachys,
 *                      which highlights the row and badges it
 *   tag       string   optional extra badge
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
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    prior_label: "Marshall prior",
    rows: [
      {
        energy: -0.49516, display: "−0.49516(1)", wf: "CNN", kind: "NQS",
        params: "7676", prior: "Yes",
        ref: "Choo, Neupert & Carleo, Phys. Rev. B 100, 125124",
        url: "https://doi.org/10.1103/PhysRevB.100.125124", year: 2019,
      },
      {
        energy: -0.495502, display: "−0.495502(1)", wf: "PEPS + CNN", kind: "Hybrid",
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
        ref: "Wang et al., Phys. Rev. B 109, 245120",
        url: "https://doi.org/10.1103/PhysRevB.109.245120", year: 2024,
      },
      {
        energy: -0.49575, display: "−0.49575(3)", wf: "RBM", kind: "Hybrid",
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
        energy: -0.4968, display: "−0.4968(4)", wf: "RBM", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Chen, Hendry, Weinberg & Feiguin, arXiv:2206.14307",
        url: "https://arxiv.org/abs/2206.14307", year: 2022,
      },
      {
        energy: -0.49717, display: "−0.49717", wf: "CNN", kind: "NQS",
        params: "106529", prior: "Yes",
        ref: "Li et al., IEEE Trans. Parallel Distrib. Syst. 33, 2846",
        url: "https://doi.org/10.1109/TPDS.2021.3136372", year: 2022,
      },
      {
        energy: -0.497437, display: "−0.497437(7)", wf: "GCNN", kind: "NQS",
        params: "67548", prior: "No",
        ref: "Roth, Szabó & MacDonald, Phys. Rev. B 108, 054410",
        url: "https://doi.org/10.1103/PhysRevB.108.054410", year: 2023,
      },
      {
        energy: -0.49747, display: "−0.49747", wf: "CNN", kind: "NQS",
        params: "421953", prior: "Yes",
        ref: "Liang et al., Mach. Learn.: Sci. Technol. 4, 015035",
        url: "https://doi.org/10.1088/2632-2153/acc56a", year: 2023,
      },
      {
        energy: -0.49755, display: "−0.49755(1)", wf: "VMC", kind: "Variational",
        params: "5", prior: "Yes",
        ref: "Hu, Becca, Parola & Sorella, Phys. Rev. B 88, 060402",
        url: "https://doi.org/10.1103/PhysRevB.88.060402", year: 2013,
      },
      {
        energy: -0.497629, display: "−0.497629(1)", wf: "RBM + PP", kind: "Hybrid",
        params: "13132", prior: "Yes",
        ref: "Nomura & Imada, Phys. Rev. X 11, 031034",
        url: "https://doi.org/10.1103/PhysRevX.11.031034", year: 2021,
      },
      {
        energy: -0.497634, display: "−0.497634(1)", wf: "ViT", kind: "NQS",
        params: "267720", prior: "No",
        ref: "Rende et al., Commun. Phys. 7, 260",
        url: "https://doi.org/10.1038/s42005-024-01732-4", year: 2024,
        tachys: true,
      },
      {
        energy: -0.4976764, display: "−0.4976764(7)", wf: "CTWF", kind: "NQS",
        params: "", prior: "",
        ref: "Chen, Naik & Heyl, arXiv:2503.10462",
        url: "https://arxiv.org/abs/2503.10462", year: 2025,
      },
      {
        energy: -0.4976921, display: "−0.4976921(4)", wf: "CNN", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Chen & Heyl, Nat. Phys. 20, 1476",
        url: "https://doi.org/10.1038/s41567-024-02566-1", year: 2024,
      },
      {
        energy: -0.4976923, display: "−0.4976923(2)", wf: "T-MPS", kind: "Hybrid",
        params: "", prior: "Yes",
        ref: "Fan et al., arXiv:2603.14425",
        url: "https://arxiv.org/abs/2603.14425", year: 2026,
      },
      {
        energy: -0.4976939, display: "−0.4976939(2)", wf: "CNN-MPS", kind: "Hybrid",
        params: "≈ 200,000", prior: "Yes",
        ref: "Fan et al., arXiv:2603.14425",
        url: "https://arxiv.org/abs/2603.14425", year: 2026,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Square",
    size: "16 x 16",
    point: "J2/J1 = 0.5",
    sites: 256,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    prior_label: "Marshall prior",
    rows: [
      {
        energy: -0.49626, display: "−0.49626", wf: "CNN", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Li et al., IEEE Trans. Parallel Distrib. Syst. 33, 2846",
        url: "https://doi.org/10.1109/TPDS.2021.3136372", year: 2022,
      },
      {
        energy: -0.49659, display: "−0.49659", wf: "CNN", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Liang et al., Mach. Learn.: Sci. Technol. 4, 015035",
        url: "https://doi.org/10.1088/2632-2153/acc56a", year: 2023,
      },
      {
        energy: -0.4967163, display: "−0.4967163(8)", wf: "CNN", kind: "NQS",
        params: "", prior: "Yes",
        ref: "Chen & Heyl, Nat. Phys. 20, 1476",
        url: "https://doi.org/10.1038/s41567-024-02566-1", year: 2024,
      },
      {
        energy: -0.496786, display: "−0.496786(1)", wf: "T-MPS", kind: "Hybrid",
        params: "", prior: "Yes",
        ref: "Fan et al., arXiv:2603.14425",
        url: "https://arxiv.org/abs/2603.14425", year: 2026,
      },
      {
        energy: -0.4969140, display: "−0.4969140(5)", wf: "CNN-MPS", kind: "Hybrid",
        params: "≈ 200,000", prior: "Yes",
        ref: "Fan et al., arXiv:2603.14425",
        url: "https://arxiv.org/abs/2603.14425", year: 2026,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Square",
    size: "20 x 20",
    point: "J2/J1 = 0.5",
    sites: 400,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    prior_label: "Marshall prior",
    rows: [
      {
        energy: -0.496732, display: "−0.496732(1)", wf: "ViT", kind: "NQS",
        params: "", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
      {
        energy: -0.4967987, display: "−0.4967987(6)", wf: "CNN-MPS", kind: "Hybrid",
        params: "≈ 200,000", prior: "Yes",
        ref: "Fan et al., arXiv:2603.14425",
        url: "https://arxiv.org/abs/2603.14425", year: 2026,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Triangular",
    size: "18 x 18",
    point: "J2/J1 = 0",
    sites: 324,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      {
        energy: -0.5357, display: "−0.53570(1)", wf: "p-BCS", kind: "Variational",
        params: "", prior: "No",
        ref: "Yunoki & Sorella, Phys. Rev. B 74, 014408",
        url: "https://doi.org/10.1103/PhysRevB.74.014408", year: 2006,
      },
      {
        energy: -0.545413, display: "−0.545413(1)", wf: "Jastrow-Gutzwiller", kind: "Variational",
        params: "", prior: "No",
        ref: "Iqbal et al., Phys. Rev. B 93, 144411",
        url: "https://doi.org/10.1103/PhysRevB.93.144411", year: 2016,
      },
      {
        energy: -0.54542, display: "−0.54542(1)", wf: "Jastrow-Gutzwiller", kind: "Variational",
        params: "", prior: "No",
        ref: "Ghorbani, Tocchio & Becca, Phys. Rev. B 93, 085111",
        url: "https://doi.org/10.1103/PhysRevB.93.085111", year: 2016,
      },
      {
        energy: -0.5459, display: "−0.5459(1)", wf: "EPS", kind: "Tensor network",
        params: "", prior: "No",
        ref: "Mezzacapo & Cirac, New J. Phys. 12, 103039",
        url: "https://doi.org/10.1088/1367-2630/12/10/103039", year: 2010,
      },
      {
        energy: -0.54716, display: "−0.54716(3)", wf: "RVB", kind: "Variational",
        params: "", prior: "No",
        ref: "Heidarian, Sorella & Becca, Phys. Rev. B 80, 012404",
        url: "https://doi.org/10.1103/PhysRevB.80.012404", year: 2009,
      },
      {
        energy: -0.54861, display: "−0.54861(1)", wf: "RNN", kind: "NQS",
        params: "", prior: "No",
        ref: "Moss et al., arXiv:2505.20406",
        url: "https://arxiv.org/abs/2505.20406", year: 2025,
      },
      {
        energy: -0.551703, display: "−0.551703(1)", wf: "ViT", kind: "NQS",
        params: "≈ 450,000", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
      {
        energy: -0.552, display: "−0.55200(1)", wf: "Zero-variance extrapolation", kind: "Extrapolation",
        params: "", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Triangular",
    size: "24 x 24",
    point: "J2/J1 = 0",
    sites: 576,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      {
        energy: -0.545374, display: "−0.545374(2)", wf: "Jastrow-Gutzwiller", kind: "Variational",
        params: "", prior: "No",
        ref: "Iqbal et al., Phys. Rev. B 93, 144411",
        url: "https://doi.org/10.1103/PhysRevB.93.144411", year: 2016,
      },
      {
        energy: -0.5481, display: "−0.54810(1)", wf: "RNN", kind: "NQS",
        params: "", prior: "No",
        ref: "Moss et al., arXiv:2505.20406",
        url: "https://arxiv.org/abs/2505.20406", year: 2025,
      },
      {
        energy: -0.551602, display: "−0.551602(1)", wf: "ViT", kind: "NQS",
        params: "≈ 450,000", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
      {
        energy: -0.55189, display: "−0.55189(1)", wf: "Zero-variance extrapolation", kind: "Extrapolation",
        params: "", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Triangular",
    size: "30 x 30",
    point: "J2/J1 = 0",
    sites: 900,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      {
        energy: -0.54534, display: "−0.54534(1)", wf: "Jastrow-Gutzwiller", kind: "Variational",
        params: "", prior: "No",
        ref: "Iqbal et al., Phys. Rev. B 93, 144411",
        url: "https://doi.org/10.1103/PhysRevB.93.144411", year: 2016,
      },
      {
        energy: -0.54793, display: "−0.54793(1)", wf: "RNN", kind: "NQS",
        params: "", prior: "No",
        ref: "Moss et al., arXiv:2505.20406",
        url: "https://arxiv.org/abs/2505.20406", year: 2025,
      },
      {
        energy: -0.551528, display: "−0.551528(1)", wf: "ViT", kind: "NQS",
        params: "≈ 450,000", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
      {
        energy: -0.5518, display: "−0.55180(1)", wf: "Zero-variance extrapolation", kind: "Extrapolation",
        params: "", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    lattice: "Triangular",
    size: "42 x 42",
    point: "J2/J1 = 0",
    sites: 1764,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      {
        energy: -0.551429, display: "−0.551429(1)", wf: "ViT", kind: "NQS",
        params: "≈ 450,000", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
      {
        energy: -0.55171, display: "−0.55171(1)", wf: "Zero-variance extrapolation", kind: "Extrapolation",
        params: "", prior: "No",
        ref: "Viteritti, Rende, Sachdev & Carleo, arXiv:2602.02665",
        url: "https://arxiv.org/abs/2602.02665", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "16 x 16",
    point: "t′/t = 0",
    pointLabel: "Hopping",
    sites: 256,
    boundary: "Open",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.726, display: "−0.7260(2)", wf: "PEPS", kind: "Tensor network",
        params: "", prior: "",
        ref: "Liu et al., Phys. Rev. Lett. 134, 256502",
        url: "", year: 2025,
      },
      {
        energy: -0.7219, display: "−0.7219", wf: "Tensor-Backflow", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7257, display: "−0.7257", wf: "Tensor-Backflow + Lanczos", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7275, display: "−0.7275", wf: "Transformer", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., Nat. Commun. 17, 7838",
        url: "https://doi.org/10.1038/s41467-026-74028-6", year: 2026,
      },
      {
        energy: -0.7236, display: "−0.7236", wf: "SCALE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.7267, display: "−0.7267", wf: "SCALE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.728, display: "−0.7280", wf: "ACE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.7288, display: "−0.7288", wf: "ACE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "16 x 16",
    point: "t′/t = 0",
    pointLabel: "Hopping",
    sites: 256,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.7515, display: "−0.7515(1)", wf: "HFPS + sym", kind: "NQS",
        params: "", prior: "",
        ref: "Roth et al., arXiv:2511.07566",
        url: "https://arxiv.org/abs/2511.07566", year: 2025,
      },
      {
        energy: -0.7509, display: "−0.7509", wf: "Tensor-Backflow", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7552, display: "−0.7552", wf: "Tensor-Backflow + Lanczos", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7563, display: "−0.7563", wf: "Transformer", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., Nat. Commun. 17, 7838",
        url: "https://doi.org/10.1038/s41467-026-74028-6", year: 2026,
      },
      {
        energy: -0.7529, display: "−0.7529", wf: "SCALE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.756, display: "−0.7560", wf: "SCALE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.7573, display: "−0.7573", wf: "ACE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.7583, display: "−0.7583", wf: "ACE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "8 x 8",
    point: "t′/t = −0.2",
    pointLabel: "Hopping",
    sites: 64,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.7447, display: "−0.7447", wf: "Transformer", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., Nat. Commun. 17, 7838",
        url: "https://doi.org/10.1038/s41467-026-74028-6", year: 2026,
      },
      {
        energy: -0.7441, display: "−0.7441(8)", wf: "Tensor-Backflow + sym + Lanczos", kind: "Hybrid",
        params: "516,096", prior: "",
        ref: "Liang, arXiv:2609.22618",
        url: "https://arxiv.org/abs/2609.22618", year: 2026,
      },
      {
        energy: -0.75058, display: "−0.75058(1)", wf: "Pfaffian + sym", kind: "NQS",
        variance: "0.016", params: "", prior: "",
        ref: "Viteritti et al., arXiv:2604.21978",
        url: "https://arxiv.org/abs/2604.21978", year: 2026,
        tachys: true,
      },
      {
        energy: -0.7540, display: "−0.7540(4)", wf: "Zero-variance extrapolation", kind: "Extrapolation",
        params: "", prior: "",
        ref: "Viteritti et al., arXiv:2604.21978",
        url: "https://arxiv.org/abs/2604.21978", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "12 x 12",
    point: "t′/t = −0.2",
    pointLabel: "Hopping",
    sites: 144,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.7414, display: "−0.7414", wf: "Transformer", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., Nat. Commun. 17, 7838",
        url: "https://doi.org/10.1038/s41467-026-74028-6", year: 2026,
      },
      {
        energy: -0.74425, display: "−0.74425(1)", wf: "SBP", kind: "NQS",
        variance: "0.035(1)", params: "≈ 500,000", prior: "",
        ref: "Rende, Viteritti & Georges, arXiv:2608.12465",
        url: "https://arxiv.org/abs/2608.12465", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "16 x 16",
    point: "t′/t = −0.2",
    pointLabel: "Hopping",
    sites: 256,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.736, display: "−0.7360(1)", wf: "HFPS + sym", kind: "NQS",
        params: "", prior: "",
        ref: "Roth et al., arXiv:2511.07566",
        url: "https://arxiv.org/abs/2511.07566", year: 2025,
      },
      {
        energy: -0.7335, display: "−0.7335", wf: "Tensor-Backflow", kind: "Hybrid",
        variance: "0.07623", params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7386, display: "−0.7386", wf: "Tensor-Backflow + Lanczos", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2507.01856",
        url: "https://arxiv.org/abs/2507.01856", year: 2025,
      },
      {
        energy: -0.7408, display: "−0.7408(1)", wf: "Tensor-Backflow + Lanczos", kind: "Hybrid",
        params: "", prior: "",
        ref: "Liang, arXiv:2609.22618",
        url: "https://arxiv.org/abs/2609.22618", year: 2026,
      },
      {
        energy: -0.7373, display: "−0.7373", wf: "SCALE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.7405, display: "−0.7405", wf: "SCALE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.743, display: "−0.7430", wf: "ACE", kind: "NQS",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.744, display: "−0.7440", wf: "ACE + GFMC", kind: "Projector",
        params: "", prior: "",
        ref: "Gu et al., arXiv:2604.25775",
        url: "https://arxiv.org/abs/2604.25775", year: 2026,
      },
      {
        energy: -0.74411, display: "−0.74411(1)", wf: "SBP", kind: "NQS",
        variance: "0.037(1)", params: "≈ 500,000", prior: "",
        ref: "Rende, Viteritti & Georges, arXiv:2608.12465",
        url: "https://arxiv.org/abs/2608.12465", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "20 x 20",
    point: "t′/t = −0.2",
    pointLabel: "Hopping",
    sites: 400,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.74382, display: "−0.74382(1)", wf: "SBP", kind: "NQS",
        variance: "0.041(1)", params: "≈ 500,000", prior: "",
        ref: "Rende, Viteritti & Georges, arXiv:2608.12465",
        url: "https://arxiv.org/abs/2608.12465", year: 2026,
        tachys: true,
      },
    ],
  },
  {
    model: "Hubbard",
    lattice: "Square",
    size: "24 x 24",
    point: "t′/t = −0.2",
    pointLabel: "Hopping",
    sites: 576,
    boundary: "Periodic",
    quantity: "Ground-state energy per site",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨i,j⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) - t' \\sum_{\\langle\\langle i,j \\rangle\\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    extra: [
      { k: "Interaction", v: "U/t = 8" },
      { k: "Doping", v: "δ = 1/8" },
    ],
    rows: [
      {
        energy: -0.74397, display: "−0.74397(1)", wf: "SBP", kind: "NQS",
        variance: "0.054(1)", params: "≈ 500,000", prior: "",
        ref: "Rende, Viteritti & Georges, arXiv:2608.12465",
        url: "https://arxiv.org/abs/2608.12465", year: 2026,
        tachys: true,
      },
    ],
  },
];

/* ---------------------------------------------------------------------------
 * EXACT — exact ground-state energies
 * ---------------------------------------------------------------------------
 * These are not variational results, so they live on their own tab instead of
 * inside the leaderboards: every cluster here is small enough to be solved
 * exactly, and no published wave function competes on it.
 *
 * Only two-dimensional lattices with periodic boundaries in every direction are
 * listed, and energies are per site, in the conventions written in each group's
 * Hamiltonian -- the same ones the leaderboard tables use.
 *
 * Group schema
 * ------------
 *   model       string  Hamiltonian family; also the label of its pill
 *   hamiltonian string  plain-text formula -- the fallback without MathJax
 *   hamiltonianTex string the same formula as LaTeX
 *
 * Row schema
 * ----------
 *   lattice   string   lattice type, with the unit-cell extents where they are
 *                      needed to pin the cluster down ("Kagome 4x4")
 *   sites     number   number of lattice sites N
 *   params    string   the remaining couplings / fillings ("" => none)
 *   eps       string   ground-state energy per site, to 8 significant digits
 * ------------------------------------------------------------------------- */

const EXACT = [
  {
    model: "Heisenberg",
    hamiltonian: "H = J Σ⟨i,j⟩ Si·Sj",
    hamiltonianTex: "H = J \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      { lattice: "Square", sites: 16, params: "", eps: "−0.7017802" },
      { lattice: "Square", sites: 36, params: "", eps: "−0.67887215" },
      { lattice: "Square", sites: 50, params: "", eps: "−0.67510204" },
      { lattice: "Triangular", sites: 16, params: "", eps: "−0.53471968" },
      { lattice: "Triangular", sites: 36, params: "", eps: "−0.5603734" },
      { lattice: "Triangular", sites: 48, params: "", eps: "−0.55860303" },
      { lattice: "Rectangular 6×8", sites: 48, params: "", eps: "−0.67598666" },
      { lattice: "Kagome 2×3", sites: 18, params: "", eps: "−0.44712615" },
      { lattice: "Kagome 4×4", sites: 48, params: "", eps: "−0.4387039" },
      { lattice: "Shuriken", sites: 24, params: "", eps: "−0.448329" },
    ],
  },
  {
    model: "J1-J2 Heisenberg",
    hamiltonian: "H = J1 Σ⟨i,j⟩ Si·Sj + J2 Σ⟨⟨i,j⟩⟩ Si·Sj",
    hamiltonianTex: "H = J_1 \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j + J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j",
    rows: [
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.05", eps: "−0.68059004" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.1", eps: "−0.65981717" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.15", eps: "−0.63954285" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.2", eps: "−0.61987394" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.25", eps: "−0.6009545" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.3", eps: "−0.58298383" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.35", eps: "−0.56624475" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.4", eps: "−0.55114777" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.45", eps: "−0.5382998" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.5", eps: "−0.52862021" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.6", eps: "−0.52589582" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.65", eps: "−0.53938247" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.7", eps: "−0.56385812" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.75", eps: "−0.59427308" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.8", eps: "−0.6273351" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.85", eps: "−0.66171967" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.9", eps: "−0.69686563" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 0.95", eps: "−0.73249885" },
      { lattice: "Square", sites: 16, params: "J₂/J₁ = 1.0", eps: "−0.76846814" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.3", eps: "−0.56245947" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.4", eps: "−0.52974501" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.5", eps: "−0.50380965" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.6", eps: "−0.49323859" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.7", eps: "−0.53000123" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.8", eps: "−0.5864866" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 0.9", eps: "−0.64905201" },
      { lattice: "Square", sites: 36, params: "J₂/J₁ = 1", eps: "−0.71436043" },
      { lattice: "Triangular", sites: 36, params: "J₂/J₁ = 0.125", eps: "−0.515564" },
      { lattice: "Triangular", sites: 48, params: "J₂/J₁ = 0.125", eps: "−0.51731444" },
      { lattice: "Rectangular 4×6", sites: 24, params: "J₂/J₁ = 0.5", eps: "−0.52252494" },
    ],
  },
  {
    model: "Hubbard",
    hamiltonian: "H = −t Σ⟨i,j⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ni↓",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle \\sigma} (c^{\\dagger}_{i\\sigma} c_{j\\sigma} + \\mathrm{h.c.}) + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}",
    rows: [
      { lattice: "Square", sites: 16, params: "U/t = 2 · N↑ = N↓ = 4", eps: "−1.1559648" },
      { lattice: "Square", sites: 16, params: "U/t = 3.5981 · N↑ = N↓ = 4", eps: "−1.1061269" },
      { lattice: "Square", sites: 16, params: "U/t = 4 · N↑ = N↓ = 4", eps: "−1.0959311" },
      { lattice: "Square", sites: 16, params: "U/t = 6 · N↑ = N↓ = 4", eps: "−1.0562454" },
      { lattice: "Square", sites: 16, params: "U/t = 7.74264 · N↑ = N↓ = 4", eps: "−1.0318222" },
      { lattice: "Square", sites: 16, params: "U/t = 8 · N↑ = N↓ = 4", eps: "−1.0287892" },
      { lattice: "Square", sites: 16, params: "U/t = 10 · N↑ = N↓ = 4", eps: "−1.0089505" },
      { lattice: "Square", sites: 16, params: "U/t = 2 · N↑ = N↓ = 5", eps: "−1.3360594" },
      { lattice: "Square", sites: 16, params: "U/t = 2.1544 · N↑ = N↓ = 5", eps: "−1.3257647" },
      { lattice: "Square", sites: 16, params: "U/t = 3.5981 · N↑ = N↓ = 5", eps: "−1.2432273" },
      { lattice: "Square", sites: 16, params: "U/t = 4 · N↑ = N↓ = 5", eps: "−1.2238086" },
      { lattice: "Square", sites: 16, params: "U/t = 6 · N↑ = N↓ = 5", eps: "−1.1473978" },
      { lattice: "Square", sites: 16, params: "U/t = 7.74264 · N↑ = N↓ = 5", eps: "−1.1002332" },
      { lattice: "Square", sites: 16, params: "U/t = 8 · N↑ = N↓ = 5", eps: "−1.0943979" },
      { lattice: "Square", sites: 16, params: "U/t = 10 · N↑ = N↓ = 5", eps: "−1.0564725" },
    ],
  },
  {
    model: "t-V",
    hamiltonian: "H = −t Σ⟨i,j⟩ (c†i cj + h.c.) + V Σ⟨i,j⟩ ni nj",
    hamiltonianTex: "H = -t \\sum_{\\langle i,j \\rangle} (c^{\\dagger}_i c_j + \\mathrm{h.c.}) + V \\sum_{\\langle i,j \\rangle} n_i n_j",
    rows: [
      { lattice: "Square", sites: 16, params: "V/t = 0.01 · 5 fermions", eps: "−0.74875164" },
      { lattice: "Square", sites: 16, params: "V/t = 0.1 · 5 fermions", eps: "−0.73766319" },
      { lattice: "Square", sites: 16, params: "V/t = 1 · 5 fermions", eps: "−0.64004066" },
      { lattice: "Square", sites: 16, params: "V/t = 10 · 5 fermions", eps: "−0.25325345" },
      { lattice: "Square", sites: 36, params: "V/t = 0.01 · 13 fermions", eps: "−0.77592793" },
      { lattice: "Square", sites: 36, params: "V/t = 0.1 · 13 fermions", eps: "−0.75945956" },
      { lattice: "Square", sites: 36, params: "V/t = 1 · 13 fermions", eps: "−0.61326034" },
      { lattice: "Square", sites: 36, params: "V/t = 10 · 13 fermions", eps: "−0.22022295" },
    ],
  },
  {
    model: "Transverse-field Ising",
    hamiltonian: "H = J Σ⟨i,j⟩ σzi σzj + Γ Σi σxi",
    hamiltonianTex: "H = J \\sum_{\\langle i,j \\rangle} \\sigma^z_i \\sigma^z_j + \\Gamma \\sum_i \\sigma^x_i",
    rows: [
      { lattice: "Square", sites: 36, params: "Γ/J = 3", eps: "−3.2009085" },
    ],
  },
];
