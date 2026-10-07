#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Simulate synthetic scRNA-seq differentiation datasets with Splatter,
# matching the characteristics of experiments/real_data/eb/data/scRNAseq
# (10x mtx format, 5 embryoid-body timepoint samples, 33,694 genes,
# ~31k cells in total).
#
# Generates 3 differentiation scenarios, each simulated once with the noisy
# configuration (dropout + expression outliers estimated from the real data):
#   linear       A -> B                                     (2 paths in a chain)
#   bifurcation  progenitor 1 -> {fate 2, fate 3}          (3 paths)
#   tree         1 -> {2,3}, 2 -> {4,5}, 3 -> {6,7}        (7 paths)
#
# Output (10x mtx format, readable with scanpy.read_10x_mtx):
#   out_dir/<scenario>/
#     matrix.mtx, genes.tsv, barcodes.tsv   counts (rows = genes, cols = cells)
#     base_cell_means.npy                   deterministic mean expression at each
#                                           cell's path position (the manifold;
#                                           colSums == ExpLibSize, no noise)
#     cell_means.npy                        true biological cell state (gamma
#                                           around base_cell_means; before the
#                                           Poisson and dropout steps)
#     row_data.csv                          gene-level Splat factors (GeneMean,
#                                           OutlierFactor, DEFacPath*, ...)
#     labels.csv                            ground truth (see below)
#     simulation.rds                        SingleCellExperiment (counts, colData)
#   out_dir/parameters.txt                  estimated + chosen parameters
#   out_dir/params.rds                      SplatParams used for simulation
#
# The two *.npy matrices are gene x cell float64 arrays (NumPy format, Fortran
# order) aligned with matrix.mtx. They are produced by the same simulation run
# and therefore one-to-one paired with the noisy counts; no separate RNG stream
# is involved.
#
# labels.csv columns:
#   Cell, Batch           cell id, batch
#   Path                  lineage label (Path1..PathN); cells on path k are
#                         "destined for" the state at the end of path k
#   Step                  Splatter pseudotime: integer position along the
#                         cell's own path (1..path.nSteps)
#   Pseudotime            Step / nSteps for that path, in [0, 1]
#   PseudotimeGlobal      global pseudotime along the unique root->path route
#                         (branch points coincide because a child path starts
#                         exactly at the end of its parent path)
#   ExpLibSize            expected library size
#
#
# Usage:
#   Rscript experiments/synthetic_data/synthetic/data_generation/simulate_splatter.R [out_dir] [nGenes] \
#       [nCells] [estCells]
#
# Requirements (Bioconductor): splatter, SingleCellExperiment, Matrix
#   BiocManager::install(c("splatter", "SingleCellExperiment"))
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(splatter)
    library(SingleCellExperiment)
    library(Matrix)
})

set.seed(42)

# --- settings ----------------------------------------------------------------
data_dir <- "experiments/real_data/eb/data/scRNAseq"
samples  <- c("T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E")

args <- commandArgs(trailingOnly = TRUE)
out_dir     <- if (length(args) >= 1) args[1] else "experiments/synthetic_data/synthetic/data"
nGenes_sim  <- if (length(args) >= 2) as.integer(args[2]) else 10000L
nCells_sim  <- if (length(args) >= 3) as.integer(args[3]) else 20000L
est_cells   <- if (length(args) >= 4) as.integer(args[4]) else 10000L



# Differential expression strength along paths. splatEstimate does NOT estimate
# these parameters (it has no group/path information), so they must be chosen
# manually.
de.prob     <- 0.5
de.facLoc   <- 1.0
de.facScale <- 0.4


# Path topologies. path.from[k] = path that path k branches from (0 = origin).
# A child path starts exactly at the END of its parent path, so:
#   linear       Path2 continues where Path1 ends
#   bifurcation  Paths 2 & 3 both branch off the end of Path1
#   tree         Paths 4 & 5 off Path2, Paths 6 & 7 off Path3
scenarios <- list(
    linear = list(
        group.prob = c(0.4, 0.6),        # fraction of cells per path
        path.from  = c(0, 1)
    ),
    bifurcation = list(
        group.prob = c(0.2, 0.4, 0.4),
        path.from  = c(0, 1, 1)
    ),
    tree = list(
        group.prob = rep(1, 7) / 7,      # 1 -> {2,3}, 2 -> {4,5}, 3 -> {6,7}
        path.from  = c(0, 1, 1, 2, 2, 3, 3)
    )
)

# --- read the real data (pool the 5 timepoint samples) -----------------------
read_10x_sample <- function(dir) {
    mat <- readMM(file.path(dir, "matrix.mtx"))
    genes <- read.delim(file.path(dir, "genes.tsv"), header = FALSE)
    barcodes <- read.delim(file.path(dir, "barcodes.tsv"), header = FALSE)
    rownames(mat) <- make.unique(genes[[1]])
    colnames(mat) <- paste0(basename(dir), "_", barcodes[[1]])
    as(mat, "CsparseMatrix")
}

cat("Loading real data...\n")
counts <- do.call(cbind, lapply(file.path(data_dir, samples), read_10x_sample))
cat(sprintf("Real data: %d genes x %d cells\n", nrow(counts), ncol(counts)))

if (est_cells > 0 && est_cells < ncol(counts)) {
    counts <- counts[, sample.int(ncol(counts), est_cells)]
    cat(sprintf("Estimating on a random subset of %d cells\n", est_cells))
}
counts <- as.matrix(counts)

# --- estimate parameters from the real data ----------------------------------
# Estimates: gamma gene means, log-normal library sizes, expression outlier
# probability/scale, BCV (edgeR), and the dropout logistic (mid + shape).
# NOTE: dropout.type is left as "none" after estimation; we switch it to
# "experiment" (with the estimated dropout curve) for the simulations below.
cat("Estimating Splat parameters from real data...\n")
params <- splatEstimate(counts)
rm(counts)
invisible(gc())

if (is.finite(mem.maxVSize()) && mem.maxVSize() < 32768) mem.maxVSize(32768)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(params, file.path(out_dir, "params.rds"))
sink(file.path(out_dir, "parameters.txt"))
cat("=== Estimated parameters (from real data) ===\n")
print(params)
cat("\n=== Simulation settings ===\n")
cat(sprintf("nGenes_sim = %d, nCells_sim = %d\n", nGenes_sim, nCells_sim))
cat(sprintf("de.prob = %g, de.facLoc = %g, de.facScale = %g, de.downProb = 0.5\n",
            de.prob, de.facLoc, de.facScale))
for (scn in names(scenarios)) {
    cat(sprintf("scenario '%s': group.prob = %s, path.from = %s, seed = %d\n",
                scn, paste(scenarios[[scn]]$group.prob, collapse = ","),
                paste(scenarios[[scn]]$path.from, collapse = ","),
                1000 + match(scn, names(scenarios)) * 10))
}
cat("\n=== Export design ===\n")
cat("Each scenario is simulated once (dropout.type = 'experiment', out.prob as\n")
cat("estimated) and exported as a noisy counts matrix plus the clean targets\n")
cat("base_cell_means.npy (deterministic manifold) and cell_means.npy (biological\n")
cat("cell states), both from the same run.\n")
cat("\n=== Session info ===\n")
print(sessionInfo())
sink()

# --- simulate ----------------------------------------------------------------
depth_from <- function(path.from) {
    # depth of each path in the tree (path 1 sits at depth 0)
    depth <- integer(length(path.from))
    for (k in seq_along(path.from)[-1]) depth[k] <- depth[path.from[k]] + 1
    depth
}


write_npy <- function(x, path) {
    x <- as.matrix(x)
    stopifnot(is.numeric(x))
    hdr <- sprintf(
        "{'descr': '<f8', 'fortran_order': True, 'shape': (%d, %d), }",
        nrow(x), ncol(x)
    )

    pad <- (64L - (10L + nchar(hdr, type = "bytes") + 1L) %% 64L) %% 64L
    hdr <- paste0(hdr, strrep(" ", pad), "\n")
    con <- file(path, "wb")
    on.exit(close(con), add = TRUE)
    writeBin(charToRaw("\x93NUMPY"), con)
    writeBin(as.raw(c(1L, 0L)), con)
    writeBin(nchar(hdr, type = "bytes"), con, size = 2L, endian = "little")
    writeBin(charToRaw(hdr), con)
    writeBin(as.vector(x), con, size = 8L, endian = "little")
}

for (scn in names(scenarios)) {
    sc <- scenarios[[scn]]
    cat(sprintf("\n=== Simulating %s ===\n", scn))

    # deterministic, distinct seed per scenario
    seed <- 1000 + match(scn, names(scenarios)) * 10

    sim <- splatSimulatePaths(
        params,
        nGenes       = nGenes_sim,
        batchCells   = nCells_sim,
        group.prob   = sc$group.prob,
        path.from    = sc$path.from,
        de.prob      = de.prob,
        de.facLoc    = de.facLoc,
        de.facScale  = de.facScale,
        dropout.type = "experiment",
        out.prob     = getParam(params, "out.prob"),
        seed         = seed,
        verbose      = TRUE
    )

    # --- ground truth labels --------------------------------------------
    cd <- as.data.frame(colData(sim))
    sim_params <- metadata(sim)$Params
    nsteps <- getParam(sim_params, "path.nSteps")
    path_id <- as.integer(cd$Group)               # 1..nPaths
    step_frac <- cd$Step / nsteps[path_id]
    depth <- depth_from(getParam(sim_params, "path.from"))

    labels <- data.frame(
        Cell              = cd$Cell,
        Batch             = cd$Batch,
        Path              = as.character(cd$Group),
        Step              = cd$Step,
        Pseudotime        = round(step_frac, 4),
        PseudotimeGlobal  = round(depth[path_id] + step_frac, 4),
        ExpLibSize        = cd$ExpLibSize
    )

    # --- export -----------------------------------------------------------
    out <- file.path(out_dir, scn)
    dir.create(out, recursive = TRUE, showWarnings = FALSE)

    cnt <- as(counts(sim), "CsparseMatrix")
    writeMM(cnt, file.path(out, "matrix.mtx"))

    write.table(
        data.frame(rowData(sim)$Gene, rowData(sim)$Gene),
        file = file.path(out, "genes.tsv"), sep = "\t",
        row.names = FALSE, col.names = FALSE, quote = FALSE
    )
    writeLines(cd$Cell, file.path(out, "barcodes.tsv"))
    write.csv(labels, file.path(out, "labels.csv"), row.names = FALSE)

    write_npy(assay(sim, "BaseCellMeans"), file.path(out, "base_cell_means.npy"))
    write_npy(assay(sim, "CellMeans"), file.path(out, "cell_means.npy"))
    write.csv(as.data.frame(rowData(sim)), file.path(out, "row_data.csv"),
              row.names = FALSE)

    saveRDS(minimiseSCE(sim, colData.keep = TRUE, assays.keep = "counts"),
            file.path(out, "simulation.rds"))

    cat(sprintf("Wrote %s (%d genes x %d cells)\n", out, nrow(cnt), ncol(cnt)))
}

cat("\nDone!\n")
