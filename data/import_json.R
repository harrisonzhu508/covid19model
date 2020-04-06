library(RJSONIO)
library(tibble)

eu_countries_11 <- c(
  "DK", # Denmark
  "IT", # Italy
  "DE", # Germany
  "ES", # Spain
  "GB", # UK
  "FR", # France
  "NO", # Norway
  "BE", # Belgium
  "AT", # Austria
  "SE", # Sweden
  "CH"  # Switzerland
)

date <- "2020-03-29"

json_to_timeseries <- function(i, countries, states) {
  country <- countries[i]
  state <- states[i]
  
  # URL from Michael's repo
  json_url <- paste0("https://raw.githubusercontent.com/MJHutchinson/google_mobility_covid19/master/data/", 
                     date, "/", state, "/2020-03-29_", country, "_", state, "_aggregate_only.json")
  json <- RJSONIO::fromJSON(json_url, flatten = TRUE)
  
  time_series <- as.data.frame(do.call(rbind, json[4:46]))
  names(time_series) <- json[[3]]
  
  time_series <- tibble::rownames_to_column(time_series, var = "date")
  time_series$date <- as.Date(time_series$date)
  
  list(country = country, state = state, data = time_series)
}

# Because these are the country wide summaries the country and state are the same
output <- lapply(1:length(eu_countries_11), 
                 json_to_timeseries,
                 countries = eu_countries_11, 
                 states = eu_countries_11)

saveRDS(output, "eu_google_mobility.rds")

# For doing US states later
# us_states_url <- "https://raw.githubusercontent.com/MJHutchinson/google_mobility_covid19/master/states.txt"
# us_states <- scan(states_url, character(), quote = "")