# ============================================================================ #
#  Code to plot results               #
# ============================================================================ #

# 0. Packages -----------------------------------------------------------------
if (!requireNamespace("tidyverse", quietly = TRUE))
  install.packages("tidyverse")
if (!requireNamespace("ggpattern", quietly = TRUE))
  install.packages("ggpattern")
if (!requireNamespace("kableExtra", quietly = TRUE))
  install.packages("kableExtra")

library(tidyverse)
library(ggpattern)
library(kableExtra)


############################################################################
# ─────────────────────────────────────────────────────────────────────────────
#  1. USER SETTINGS                                                            #
# ─────────────────────────────────────────────────────────────────────────────

# adjust the folder the results are saved
# under that folder you have GRID and RNMD folders.

simPath <- "./"
simPrefix <- "RND"


resultsPath <- paste0(simPath, simPrefix)

# cat(resultsPath)

setwd(resultsPath)

plotWidth <- 30 # cm
plotHeight <- plotWidth / 1.618 #cm

# --- 1) Define acronyms (LaTeX-safe; escape=FALSE is already set) -----------
algo_acronym <- c(
#  "Random Walk"                      = "RW",
#  "Plain QLearning"                  = "QL",
  "Adaptive ε"                       = "AQ",
  "Pruning V1-1.1"                   = "P1-1.1",
  "Pruning V1-1.6"                   = "P1-1.6",
  "Pruning V2-1.1"                   = "P2-1.1",
  "Pruning V2-1.6"                   = "P2-1.6",
  "Adaptive ε + Pruning V1-1.6"      = "AQ+P1-1.6",
  "Adaptive ε + Pruning V1-1.1"      = "AQ+P1-1.1",
  "Adaptive ε + Pruning V2-1.6"      = "AQ+P2-1.6",
  "Adaptive ε + Pruning V2-1.1"      = "AQ+P2-1.1"
)

# Optional: keep a separate display order (decoupled from file_map)
algo_display_order <- c(
  "Adaptive ε + Pruning V2-1.1",
  "Adaptive ε + Pruning V2-1.6",
  "Adaptive ε + Pruning V1-1.1",
  "Adaptive ε + Pruning V1-1.6",
  "Pruning V2-1.1",
  "Pruning V2-1.6",
  "Pruning V1-1.1",
  "Pruning V1-1.6",
  "Adaptive ε"
#  "Plain QLearning",
#  "Random Walk"
)




file_map <- c(
#  "Random Walk"          = "summary_rnd-wlk-rnd.csv",
#  "Plain QLearning"      = "summary_ql-rnd.csv",
  "Adaptive ε"           = "summary_adaptive-rnd.csv",
  "Pruning V1-1.1"              = "summary_pruning-v2-rnd-backoff-1.1.csv",
  "Pruning V1-1.6"              = "summary_pruning-v2-rnd-backoff-1.6.csv",
  "Pruning V2-1.1"              = "summary_pruning-v4-rnd-backoff-1.1.csv",
  "Pruning V2-1.6"              = "summary_pruning-v4-rnd-backoff-1.6.csv",
  "Adaptive ε + Pruning V1-1.6" = "summary_adaptive-pruning-v2-rnd-backoff-1.6.csv",
  "Adaptive ε + Pruning V1-1.1" = "summary_adaptive-pruning-v2-rnd-backoff-1.1.csv",
  "Adaptive ε + Pruning V2-1.6" = "summary_adaptive-pruning-v4-rnd-backoff-1.6.csv",
  "Adaptive ε + Pruning V2-1.1" = "summary_adaptive-pruning-v4-rnd-backoff-1.1.csv"
)


# sanity checks (helps catch typos)
stopifnot(setequal(names(file_map), names(algo_acronym)))
stopifnot(setequal(algo_display_order, names(file_map)))


############################################################################
# ─────────────────────────────────────────────────────────────────────────────
#  0.  BUILD ONE RAW DATAFRAME FROM ALL SIMULATION txt FILES                  #
# ─────────────────────────────────────────────────────────────────────────────
# file_map MUST already exist, e.g.
folder_map <- c(
#  "Random Walk" = "rnd-wlk",
#  "Plain QLearning" = "ql",
  "Adaptive ε" = "adaptive",
  "Pruning V1-1.6" = "pruning-v2-backoff-1.6",
  "Pruning V1-1.1" = "pruning-v2-backoff-1.1",
  "Pruning V2-1.6" = "pruning-v4-backoff-1.6",
  "Pruning V2-1.1" = "pruning-v4-backoff-1.1",
  "Adaptive ε + Pruning V1-1.1" = "adaptive-pruning-v2-backoff-1.1",
  "Adaptive ε + Pruning V1-1.6" = "adaptive-pruning-v2-backoff-1.6",
  "Adaptive ε + Pruning V2-1.6" = "adaptive-pruning-v4-backoff-1.6",
  "Adaptive ε + Pruning V2-1.1" = "adaptive-pruning-v4-backoff-1.1")




############################################################################

read_sim <- function(path, algo_label) {
  # pattern: <algoname>-out<N>-<sim>.txt  (sim = 1…20)
  parts <- stringr::str_match(basename(path),
                              "^(.*)-out(\\d+)-(\\d+)\\.txt$")
  Nval  <- as.integer(parts[3])
  simid <- as.integer(parts[4])
  
  read_table(path,
             # In case of ql, rnd-wlk:
             # col_names = c("A","G","E","t","edge","Nfile"),
             col_names = c("A","G","E","AVGE","t","edge","Nfile","TotEdge"),
            
             show_col_types = FALSE) %>%
    mutate(
      Algorithm = algo_label,
      N         = Nval,
      SimID     = simid
    )
}

raw_all_df <- purrr::imap(folder_map, function(folder, algo_label) {
  files <- list.files(path = folder,
                      pattern = "\\.txt$",
                      recursive = FALSE,
                      full.names = TRUE)
  purrr::map_dfr(files, read_sim, algo_label = algo_label)
}) %>% list_rbind()

saveRDS(raw_all_df, "raw_all.rds")        # use later:  raw_df <- readRDS("raw_all.rds")




## 1b) metrics to iterate over (order matters)
metric_vec <- c("AVGTime", "AVGEdgePercent", "EdgeExcessPercent", "EPTU")

order_mode    <- "fixed"   # or "rank"
use_hatch     <- TRUE
legend_where  <- "top"
x_axis_label  <- "N  (for an N×N antennas)"

pattern_vals <- c("stripe","crosshatch","circle","grid","none","wave")

############################################################################
# ─────────────────────────────────────────────────────────────────────────────
#  2. HELPERS                                                                  #
# ─────────────────────────────────────────────────────────────────────────────
print_latex <- function(df, caption, file = NULL, digits = 3) {
  tab <- knitr::kable(df, format = "latex", booktabs = TRUE,
                      digits = digits, caption = caption, align = "c") |>
    kableExtra::kable_styling(position = "center",
                              latex_options = c("hold_position"))
  if (!is.null(file)) writeLines(tab, file)
  cat(tab, sep = "\n")
}




# ---------- Upper-triangle % improvement matrix (column vs row) ----------
make_upper_triangle_delta <- function(wide, algo_levels, higher_is_better = TRUE,
                                      digits = 3, diagonal = "\\textemdash{}",
                                      show_sign = FALSE) {
  m   <- length(algo_levels)
  # For lower-is-better metrics, flip sign so that "positive = column better than row"
  #dir <- if (higher_is_better) 1 else -1
  
  M <- matrix(NA_real_, nrow = m, ncol = m,
              dimnames = list(algo_levels, algo_levels))
  
  for (i in seq_len(m)) {
    for (j in seq_len(m)) {
      if (i == j) {
        M[i, j] <- NA_real_
      } else if (j > i) {
        old <- wide[[algo_levels[i]]]  # row (baseline)
        new <- wide[[algo_levels[j]]]  # column (candidate)
        # % improvement per-N: 100 * (new - old) / old
        #pct <- 100 * (new - old) / old
        
        #if (higher_is_better) pct <- 100 * (new - old) / new
        #else pct <- 100 * (new - old) / old
        #dir <- 1
        if (higher_is_better) pct <- old / new
        else pct <- new / old
        # Guard against divide-by-zero/Inf/NaN
        pct[!is.finite(pct)] <- NA_real_
        M[i, j] <- mean(pct, na.rm = TRUE)
      } else {
        M[i, j] <- NA_real_
      }
    }
  }
  
  fmt_num <- function(x) {
    if (is.na(x)) return("")
    if (show_sign) sprintf("%+.*f", digits, x) else sprintf("%.*f", digits, x)
  }
  M_chr <- apply(M, c(1, 2), fmt_num)
  diag(M_chr) <- diagonal
  M_chr
}



# ---------- Print LaTeX with kableExtra ----------
print_latex_upper_triangle <- function(M_chr, caption, file = NULL, font_size = 9) {
  tab <- knitr::kable(M_chr, format = "latex", booktabs = TRUE,
                      caption = caption, align = "c", escape = FALSE) |>
    kableExtra::kable_styling(position = "center",
                              latex_options = c("hold_position", "scale_down"),
                              font_size = font_size) |>
    kableExtra::add_header_above(
      c(" " = 1, "Comparison matrix (column − row)" = ncol(M_chr) - 1)
    )
  latex_out <- as.character(tab)
  if (!is.null(file)) writeLines(latex_out, file)
  cat(latex_out, sep = "\n")
}






summarise_file <- function(path, label) {
  read_csv(path, show_col_types = FALSE) |>
    transmute(N, TotEdge, AVGTime = t_mean, EdgeMean = edge_mean) |>
    group_by(N) |>
    summarise(
      TotEdge  = dplyr::first(TotEdge),
      AVGTime  = mean(AVGTime,  na.rm = TRUE),
      EdgeMean = mean(EdgeMean, na.rm = TRUE),
      .groups  = "drop"
    ) |>
    mutate(
      AVGEdgePercent      = 100 * EdgeMean / TotEdge,
      SpanningTreePercent = 100 * (N*N - 1) / TotEdge,
      EdgeExcessPercent   = AVGEdgePercent - SpanningTreePercent,
      EPTU                = EdgeMean / AVGTime,
      Algorithm           = label
    )
}



# helper (near your user settings)
get_pattern_vals <- function(n) {
  base <- c("stripe","crosshatch","circle","grid","none","wave",
            "weave","horizontal","vertical","dot","plus","zigzag")
  rep_len(base, n)
}

plot_metric <- function(df, metric, higher_is_better) {
  # drop rows without the metric and drop unused algo levels
  df <- df %>% filter(!is.na(.data[[metric]]))
  df$Algorithm <- droplevels(df$Algorithm)
  
  algos <- levels(df$Algorithm)
  n_alg <- length(algos)
  
  # --- colors: Set2 up to 8, otherwise switch to hue palette
  fill_vals <- setNames(
    if (n_alg <= 8)
      scales::brewer_pal(type = "qual", palette = "Set2")(n_alg)
    else
      scales::hue_pal()(n_alg),
    algos
  )
  
  p <- ggplot(df, aes(factor(N), .data[[metric]],
                      fill = Algorithm,
                      pattern = if (use_hatch) Algorithm)) +
    geom_col(position = position_dodge(.8), colour = "black",
             alpha = if (use_hatch) .85 else 1)
  
  p <- p + scale_fill_manual(values = fill_vals, name = "[Color - Algorithm]:")
  
  if (use_hatch) {
    pat_vals <- setNames(get_pattern_vals(n_alg), algos)
    p <- p + ggpattern::scale_pattern_manual(values = pat_vals, name = "Algorithm")
  }
  
  p <- p +
    labs(x = x_axis_label, y = metric) +
    theme_bw(base_size = 12) +
    theme(legend.position = legend_where,
          legend.title = element_text(face = "bold"))
  
  # ... keep your ggsave calls
  
  ggsave(filename = paste0(simPrefix, "-", metric,".png"),
         width  = plotWidth,
         height = plotHeight,
         units  = "cm",
         dpi    = 300,
         # width = plotWidth, 
         # height = plotWidth, 
         # # pointsize = 14, 
         # units = "in", 
         # dpi=300
  )
  # 
  # ggsave(filename = paste0(simPrefix,  "-", metric,".eps"),
  #        device = cairo_ps,          # <- cairographics
  #        fallback_resolution = 300,  # rasterises only what PS can't draw
  #        onefile = FALSE,
  #        width  = plotWidth,
  #        height = plotHeight,
  #        units  = "cm",
  #        dpi    = 300
  #        
  #        # width = plotWidth, 
  #        # height = plotWidth, 
  #        # # pointsize = 14, 
  #        # units = "in", 
  #        # dpi=300
  # )
  
  
  p
}






############################################################################
# ─────────────────────────────────────────────────────────────────────────────
#  3. READ & PREPARE DATA (once)                                               #
# ─────────────────────────────────────────────────────────────────────────────
results <- purrr::imap(file_map, summarise_file) |> list_rbind()

# ─────────────────────────────────────────────────────────────────────────────
#  4. LOOP OVER METRICS                                                       #
# ─────────────────────────────────────────────────────────────────────────────
for (metric_choice in metric_vec) {
  
  higher_is_better <- metric_choice %in% c("AVGEdgePercent", "EdgeExcessPercent", "EPTU")
 
  
  
  
  
  # 4a) choose order ---------------------------------------------------------
  algo_levels <- if (order_mode == "fixed") {
    algo_display_order
  } else {
    results |>
      dplyr::group_by(Algorithm) |>
      dplyr::summarise(val = mean(.data[[metric_choice]], na.rm = TRUE), .groups="drop") |>
      dplyr::arrange(if (higher_is_better) dplyr::desc(val) else val) |>
      dplyr::pull(Algorithm)
  }
  
  df_metric <- results %>%
    dplyr::mutate(Algorithm = factor(Algorithm, levels = algo_levels))
  
  
  
  
  # 4b) plot -----------------------------------------------------------------
  print(plot_metric(df_metric, metric_choice, higher_is_better))
  
  
  
  
  # keep wide columns ordered the same as algo_levels
  wide <- df_metric %>%
    dplyr::group_by(N, Algorithm) %>%
    dplyr::summarise(val = mean(.data[[metric_choice]], na.rm = TRUE), .groups="drop") %>%
    dplyr::mutate(Algorithm = factor(Algorithm, levels = algo_levels)) %>%
    dplyr::arrange(N, Algorithm) %>%
    tidyr::pivot_wider(names_from = Algorithm, values_from = val) %>%
    dplyr::select(N, dplyr::all_of(algo_levels))
  
  # ---- build matrix (using full names to compute) ----------------------------
  M_chr <- make_upper_triangle_delta(
    wide             = wide,
    algo_levels      = algo_levels,
    higher_is_better = higher_is_better,
    digits           = 3,
    diagonal         = "\\textemdash{}",
    show_sign        = FALSE
  )
  
  # ---- swap to display labels: "ACR (Full Name)" -----------------------------
  # display_labels <- paste0(unname(algo_acronym[algo_levels]), " (", algo_levels, ")")
  # Alternative styles:
  display_labels <- unname(algo_acronym[algo_levels])          # acronym only
  # display_labels <- paste0(algo_levels, " [", unname(algo_acronym[algo_levels]), "]")
  
  dimnames(M_chr) <- list(display_labels, display_labels)
  
  print_latex_upper_triangle(
    M_chr,
    caption = paste(
      "Upper-triangle pairwise average delta for", metric_choice,
      "(positive: column better than row)"
    ),
    file = paste0(simPrefix, "-matrix_", metric_choice, ".tex"),
    font_size = 8
  )
  
  
  
  
  
}


# Build once (use your chosen display order)
legend_tbl <- tibble::tibble(
  Acronym   = unname(algo_acronym[algo_display_order]),
  Algorithm = algo_display_order
)

# 1) LaTeX table -> .tex  (keeps LaTeX like A$\\varepsilon$)
legend_tex <- knitr::kable(
  legend_tbl, format = "latex", booktabs = TRUE, align = "l",
  caption = "Algorithm acronyms used in comparison matrices.",
  escape = FALSE
) |>
  kableExtra::kable_styling(position = "center",
                            latex_options = c("hold_position", "scale_down"),
                            font_size = 9)

writeLines(as.character(legend_tex), paste0(simPrefix, "-legend_acronyms.tex"))

# 2) Plain text (tab-separated) -> .txt
# readr::write_tsv(legend_tbl, paste0(simPrefix, "-legend_acronyms.txt"))

# (Optional) CSV too
# readr::write_csv(legend_tbl, paste0(simPrefix, "-legend_acronyms.csv"))



############################################################################
# Scatter Plots for A, G, E permutations
############################################################################


# 5a) read raw data WITH α,γ,ε columns (no aggregation) ----------------------
read_raw <- function(path, label) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(Algorithm = label)
}
raw_df <- purrr::imap(file_map, read_raw) %>% list_rbind()

# 5b) build a combo index (1…125) per Algorithm × N --------------------------
raw_df <- raw_df %>%
  group_by(Algorithm, N) %>%
  mutate(combo_id = row_number()) %>%
  ungroup()

# 5c) loop over grid sizes ---------------------------------------------------
  
plot_combo_scatter <- function(
    raw_df,
    combo_order   = c("A","G","AVGE"),
    simPrefix     = get0("simPrefix", inherits = TRUE, ifnotfound = "RND"),
    base_algos    = c("Random Walk", "Plain QLearning", "Pruning V1-1.1","Pruning V1-1.6", "Pruning V2-1.1","Pruning V2-1.6"),
    include_algos = NULL,
    target_labels = 20,
    width_cm      = 40,
    aspect_phi    = 1.618
) {
  stopifnot(length(combo_order) == 3, setequal(combo_order, c("A","G","AVGE")))
  file_suffix <- paste0(combo_order, collapse = "")
  
  # If user passed a subset, keep only those algos (in that order)
  if (!is.null(include_algos)) {
    raw_df <- raw_df |> dplyr::filter(Algorithm %in% include_algos)
    raw_df$Algorithm <- factor(raw_df$Algorithm, levels = include_algos)
    # keep base_algos that remain after filtering
    base_algos <- intersect(base_algos, include_algos)
  }
  
  
  
  # helpers
  fmt_num <- function(x, digits = 3) {
    out <- ifelse(is.na(x), "–", formatC(x, digits = digits, format = "fg"))
    as.character(out)
  }
  mode_dbl <- function(x) {
    x <- x[!is.na(x)]
    if (!length(x)) return(NA_real_)
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  out_files <- character(0)
  
  for (n_val in sort(unique(raw_df$N))) {
    
    dfN <- raw_df %>%
      dplyr::filter(N == n_val) %>%
      dplyr::mutate(Algorithm = factor(Algorithm))
    
    # canonical (A,G,E) per combo_id
    upd_key <- dfN %>%
      dplyr::filter(!Algorithm %in% base_algos) %>%
      dplyr::group_by(combo_id) %>%
      dplyr::summarise(
        A_upd = dplyr::first(na.omit(A)),
        G_upd = dplyr::first(na.omit(G)),
        AVGE_upd = mode_dbl(AVGE),
        .groups = "drop"
      )
    
    base_key <- dfN %>%
      dplyr::filter(Algorithm %in% base_algos) %>%
      dplyr::group_by(combo_id) %>%
      dplyr::summarise(
        A_base = dplyr::first(na.omit(A)),
        G_base = dplyr::first(na.omit(G)),
        AVGE_base = dplyr::first(na.omit(AVGE)),
        .groups = "drop"
      )
    
    labels_tbl <- dfN %>%
      dplyr::distinct(combo_id) %>%
      dplyr::left_join(upd_key,  by = "combo_id") %>%
      dplyr::left_join(base_key, by = "combo_id") %>%
      dplyr::transmute(
        combo_id,
        A = dplyr::coalesce(A_upd, A_base),
        G = dplyr::coalesce(G_upd, G_base),
        AVGE = dplyr::coalesce(AVGE_upd, AVGE_base)
      )
    
    # order by combo_order; NAs last
    labels_tbl_ord <- labels_tbl %>%
      dplyr::mutate(
        A_ord = ifelse(is.na(A), Inf, A),
        G_ord = ifelse(is.na(G), Inf, G),
        AVGE_ord = ifelse(is.na(AVGE), Inf, AVGE)
      ) %>%
      dplyr::arrange(dplyr::across(dplyr::all_of(paste0(combo_order, "_ord")))) %>%
      dplyr::mutate(combo_idx = dplyr::row_number())
    
    # labels in the chosen order
    lab_vals <- labels_tbl_ord %>%
      dplyr::transmute(
        combo_id, combo_idx,
        v1 = fmt_num(.data[[combo_order[1]]]),
        v2 = fmt_num(.data[[combo_order[2]]]),
        v3 = fmt_num(.data[[combo_order[3]]])
      ) %>%
      dplyr::mutate(combo_lab = sprintf("(%s,%s,%s)", v1, v2, v3)) %>%
      dplyr::select(combo_id, combo_idx, combo_lab)
    
    # join x-position + label
    dfN <- dfN %>%
      dplyr::left_join(lab_vals, by = "combo_id") %>%
      dplyr::mutate(
        combo_lab = dplyr::if_else(Algorithm == "Random Walk", "(–,–,–)", combo_lab)
      )
    
    # split datasets
    df_upd  <- dfN %>% dplyr::filter(!Algorithm %in% base_algos)
    df_base <- dfN %>% dplyr::filter( Algorithm %in% base_algos)
    
    # extremes
    mins <- dfN %>% dplyr::group_by(Algorithm) %>% dplyr::slice_min(t_mean, n = 1, with_ties = FALSE)
    maxs <- dfN %>% dplyr::group_by(Algorithm) %>% dplyr::slice_max(t_mean, n = 1, with_ties = FALSE)
    
    # colors
    lvls <- levels(dfN$Algorithm)
    manual_cols <- c("pink","orange","gray40","purple","dodgerblue","darkgreen","black","brown","lightgreen", "yellow", "red")
    if (length(lvls) > length(manual_cols)) stop("Need more colours! Extend manual_cols.")
    algo_pal <- stats::setNames(manual_cols[seq_along(lvls)], lvls)
    
    # shapes
    base_shape_map <- c(
      "Random Walk"        = 24,
      "Plain QLearning"    = 22,
      "Pruning1.6"            = 23,
      "Adaptive QLearning" = 21,
      "Pruning1.1"    = 25,
      "Softmax"            = 3
    )
    shape_pool <- c(21,22,23,24,25,3,4,8,19,7)
    shape_map <- as.numeric(base_shape_map[lvls]); names(shape_map) <- lvls
    used <- stats::na.omit(shape_map); pool <- setdiff(shape_pool, used)
    if (any(is.na(shape_map))) {
      shape_map[is.na(shape_map)] <- pool[seq_len(sum(is.na(shape_map)))]
    }
    
    # x-axis ticks
    upd_idxs <- sort(unique(df_upd$combo_idx))
    all_idxs <- sort(unique(dfN$combo_idx))
    xmin_all <- min(all_idxs, na.rm = TRUE)
    xmax_all <- max(all_idxs, na.rm = TRUE)
    lab_map_idx <- stats::setNames(lab_vals$combo_lab, lab_vals$combo_idx)
    
    base_pool_ids <- if (length(upd_idxs)) upd_idxs else all_idxs
    tick_step <- max(1, ceiling(length(base_pool_ids) / target_labels))
    break_ids <- base_pool_ids[seq(1, length(base_pool_ids), by = tick_step)]
    break_labels <- unname(lab_map_idx[as.character(break_ids)])
    
    title_expr <- bquote(
      "Mean discovery time vs. (" * .(combo_order[1]) * "," * .(combo_order[2]) * "," * .(combo_order[3]) *
        ") combinations (N =" ~ .(n_val) * ")"
    )
    
    # plot
    p_scatter <-
      ggplot2::ggplot() +
      ggplot2::geom_line(
        data = df_upd,
        ggplot2::aes(combo_idx, t_mean, colour = Algorithm, group = Algorithm),
        alpha = 0.35, linewidth = 0.4
      ) +
      ggplot2::geom_point(
        data = df_upd,
        ggplot2::aes(combo_idx, t_mean, colour = Algorithm, shape = Algorithm, fill = Algorithm),
        size = 2.0, stroke = 0.25, alpha = 0.9
      ) +
      ggplot2::geom_line(
        data = df_base,
        ggplot2::aes(combo_idx, t_mean, colour = Algorithm, group = Algorithm),
        linewidth = 0.6
      ) +
      ggplot2::geom_point(
        data = df_base,
        ggplot2::aes(combo_idx, t_mean, colour = Algorithm, shape = Algorithm, fill = Algorithm),
        size = 2.4, stroke = 0.35
      ) +
      ggplot2::geom_point(
        data = df_base %>% dplyr::filter(Algorithm == "Random Walk"),
        ggplot2::aes(combo_idx, t_mean),
        shape = 24, size = 2.8, stroke = 0.5, fill = "white", colour = "black"
      ) +
      ggplot2::geom_point(data = mins, ggplot2::aes(combo_idx, t_mean), shape = 21, fill = NA, colour = "green", stroke = 1.0, size = 3.6) +
      ggplot2::geom_point(data = maxs, ggplot2::aes(combo_idx, t_mean), shape = 21, fill = NA, colour = "red",   stroke = 1.0, size = 3.6) +
      ggplot2::scale_colour_manual(values = algo_pal, name = "Algorithm") +
      ggplot2::scale_shape_manual(values = shape_map, name = "Algorithm") +
      ggplot2::scale_fill_manual(values = algo_pal, guide = "none") +
      ggplot2::scale_x_continuous(
        limits = c(xmin_all, xmax_all),
        breaks = break_ids,
        labels = break_labels,
        expand = ggplot2::expansion(mult = 0.01)
      ) +
      ggplot2::labs(
        title = title_expr,
        x = NULL, y = "mean  t  [time-units]"
      ) +
      ggplot2::theme_bw(base_size = 12) +
      ggplot2::theme(
        legend.position = "bottom",
        panel.grid.minor.x = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_text(angle = 80, hjust = 1, vjust = 1, size = 6)
      )
    
    fPrefix <- paste0(simPrefix, "-scatter-", file_suffix)
    out_file <- paste0(fPrefix, sprintf("-N%d.png", n_val))
    ggplot2::ggsave(out_file, p_scatter,
                    width = width_cm, height = width_cm / aspect_phi,
                    units = "cm", dpi = 300, limitsize = FALSE)
    out_files <- c(out_files, out_file)
  }
  
  invisible(out_files)
}

# # Reminder
# algo_display_order <- c(
#   "Adaptive ε + Pruning V2-1.1",
#   "Adaptive ε + Pruning V2-1.6",
#   "Adaptive ε + Pruning V1-1.1",
#   "Adaptive ε + Pruning V1-1.6",
#   "Pruning V2-1.1",
#   "Pruning V2-1.6",
#   "Pruning V1-1.1",
#   "Pruning V1-1.6",
#   "Adaptive ε"
#   #  "Plain QLearning",
#   #  "Random Walk"
# )

# ─────────────────────────────────────────────────────────────────────────────
#  5.  SCATTER PLOTS OF  t_mean  vs. A,G,E COMBINATIONS – one per N           #
# ─────────────────────────────────────────────────────────────────────────────

plot_combo_scatter(raw_df, combo_order = c("A","G","AVGE"), simPrefix="RND")
# Only show adaptive + pruning V2 variants
plot_combo_scatter(raw_df,
                   include_algos = c("Adaptive ε + Pruning V2-1.1", "Adaptive ε + Pruning V2-1.6"),
                   combo_order = c("A","G","AVGE"),
                   simPrefix="RND"
)

# Show just baselines vs Adaptive ε
plot_combo_scatter(raw_df,
                   include_algos = c("Pruning V1-1.1"),
                   combo_order = c("A","G","AVGE"),
                   simPrefix="RND"
)




# ─────────────────────────────────────────────────────────────────────────────
#  6.  SCATTER PLOTS OF  t_mean  vs. E,G,A COMBINATIONS – one per N           #
# ─────────────────────────────────────────────────────────────────────────────
plot_combo_scatter(raw_df,
                   include_algos = c("Pruning V1-1.1"),
                   combo_order = c("AVGE","G","A"),
                   simPrefix="RND"
)
plot_combo_scatter(raw_df, combo_order = c("AVGE","G","A"), simPrefix="RND")




# ─────────────────────────────────────────────────────────────────────────────
#  7.  SCATTER PLOTS OF  t_mean  vs. E,A,G COMBINATIONS – one per N           #
# ─────────────────────────────────────────────────────────────────────────────

plot_combo_scatter(raw_df, combo_order = c("AVGE","A","G"), simPrefix="RND")









############################################################################

############################################################################

############################################################################
