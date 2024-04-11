# option <- "hemibrain"
# option <- "manc"
option <- "pyc-pyc"
# network_sizes <- c(250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 17500, 20000)
network_sizes <- c(6, 10, 20, 30, 40, 50, 82, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000)

for (network_size in network_sizes){
  total_nodes <- as.character(network_size)
  # total_nodes <- "1000"
  # if total_nodes <- "None" the number of nodes is then the number of nodes for which the data was optimised
  
  
  base_filename <- "data_preparation/degree_elements/degrees_experimental_"
  csv_extension <- ".csv"
  full_experimental_filename <- paste(base_filename, option, csv_extension, sep = '')
  
  # experimental_data_dataframe <- read.csv("data_preparation\\degree_elements\\degrees_model_0.csv")
  experimental_data_dataframe <- read.csv(full_experimental_filename)
  experimental_data_indegree <- experimental_data_dataframe$indegree
  experimental_data_outdegree <- experimental_data_dataframe$outdegree
  
  base_statistics_filename <- "ks_stats/"
  full_statistics_filename <- paste(base_statistics_filename, option, "/", "ks_stats_", total_nodes, csv_extension, sep = '')
  # experimental_data_dataframe <- read.csv("data_preparation/degree_elements/indegrees_data_hemibrain.csv")
  
  base_path <- paste("data_preparation\\degree_elements\\", option, "\\", sep = '')
  search_pattern_filenames <- paste("^degrees_model_", total_nodes, "_", ".*\\.csv$", sep = '')
  model_degree_files <- list.files(path = base_path, pattern = search_pattern_filenames)
  num_files = length(model_degree_files)
  ks_stats <- data.frame(
    D_statistic_indegree = numeric(num_files),
    p_value_indegree = numeric(num_files),
    D_statistic_outdegree = numeric(num_files),
    p_value_outdegree = numeric(num_files)
  )
  
  # Iterate over each CSV file
  for (seed_number in seq_along(model_degree_files)) {
    file <- model_degree_files[seed_number]
    # Read the CSV file
    model_data_dataframe <- read.csv(paste(base_path, file, sep = ''))
    model_data_indegree <- model_data_dataframe$indegree
    model_data_outdegree <- model_data_dataframe$outdegree
    # ks_general_stats_indegree <- KSgeneral::disc_ks_test(sample(1:80, 5000, replace=TRUE),
    # ecdf(experimental_data_indegree),
    # exact = TRUE)
    ks_general_stats_indegree <- KSgeneral::disc_ks_test(model_data_indegree,
                                                         ecdf(experimental_data_indegree),
                                                         exact = TRUE)
    ks_stats$D_statistic_indegree[seed_number] <- ks_general_stats_indegree$statistic
    ks_stats$p_value_indegree[seed_number] <- ks_general_stats_indegree$p.value
    # ks_general_stats_outdegree <- KSgeneral::disc_ks_test(sample(1:80, 5000, replace=TRUE),
    # ecdf(experimental_data_outdegree),
    # exact = TRUE)
    ks_general_stats_outdegree <- KSgeneral::disc_ks_test(model_data_outdegree,
                                                          ecdf(experimental_data_outdegree),
                                                          exact = TRUE)
    ks_stats$D_statistic_outdegree[seed_number] <- ks_general_stats_outdegree$statistic
    ks_stats$p_value_outdegree[seed_number] <- ks_general_stats_outdegree$p.value
    
  }
  write.csv(ks_stats, full_statistics_filename, row.names = FALSE)
}


