// functions {
//     matrix expected_deaths(matrix prediction, matrix f) {
//         int M = rows(prediction);
//         int N = cols(prediction);
//         matrix[M, N] y;
//         for (m in 1:M) {
//             y[1, m] = 1e-9;
// 	        for (i in 2:M) {
// 	            y[2, m] = 0;
// 	            for (j in 1:(i - 1)) {
// 	                y[i, m] += prediction[j, m] * f[i - j, m];
//                 }
//             }
//         }
//         return y;
//     }
// }

data {
    int<lower = 1> M; // number of countries
    int<lower = 1> N0; // number of days for which to impute infections
    int<lower = 1> N[M]; // days of observed data for country m. each entry must be <= N2
    int<lower = 1> N2; // # days of observed data (used in analysis) + # of days to forecast
    // real<lower=0> x[N2]; // index of days (starting at 1)
    int cases[N2, M]; // reported cases
    int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
    matrix[N2, M] f; // h * s
    matrix[N2, M] intervention1;
    matrix[N2, M] intervention2;
    matrix[N2, M] intervention3;
    matrix[N2, M] intervention4;
    matrix[N2, M] intervention5;
    matrix[N2, M] intervention6;

    // mobility covariates
    matrix[N2, M] x1;
    matrix[N2, M] x2;
    matrix[N2, M] x3;
    matrix[N2, M] x4;
    matrix[N2, M] x5;
    int EpidemicStart[M];
    real SI[N2]; // fixed pre-calculated SI using emprical data from Neil
}

transformed data {
    matrix[N2, M] matrix_1000 = rep_matrix(1000, N2, M);
    matrix[N2, M] matrix_0 = rep_matrix(0, N2, M);
    real delta = 1e-5;
}

parameters {
    vector<lower = 0>[M] y_raw;
    real<lower = 0> mu[M]; // intercept for Rt
    real<lower = 0> alpha[6]; // the hier term
    real<lower = 0> kappa_std;
    real<lower = 0> phi_std;
    real<lower = 0> tau_unit;

    real<lower=0> length_scale_1;
    real<lower=0> scale_factor_1;
    real<lower=0> length_scale_2;
    real<lower=0> scale_factor_2;
    real<lower=0> length_scale_3;
    real<lower=0> scale_factor_3;
    real<lower=0> length_scale_4;
    real<lower=0> scale_factor_4;
    real<lower=0> length_scale_5;
    real<lower=0> scale_factor_5;

    vector[N2] eta;
}

transformed parameters {
    real<lower = 0> tau = tau_unit / 0.03;
    real<lower = 0> phi = 5 * phi_std;
    real<lower = 0> kappa = 0.5 * kappa_std;
    vector<lower = 0>[M] y = tau * y_raw;
    matrix[N2, M] prediction = matrix_0;
    matrix[N2, M] E_deaths  = matrix_0;
    matrix[N2, M] Rt;

    // GP formulation
    matrix[N2, N2] K;
    matrix[N2, N2] L;
    vector<lower = 0>[N2] exp_gp;
    
    Rt = exp(intervention1 * (-alpha[1]) 
            + intervention2 * (-alpha[2]) 
            + intervention3 * (-alpha[3]) 
            + intervention4 * (-alpha[4]) 
            + intervention5 * (-alpha[5]) 
            + intervention6 * (-alpha[6])); // + GP[i]); // to_vector (x) * time_effect
    for (m in 1:M) {
        for (i in 1:N0) {
            prediction[i, m] = y[m];
        } // learn the number of cases in the first N0 days
        Rt[, m] *= mu[m];
        
        // GP
        K = cov_exp_quad(to_array_1d(x1[, m]), scale_factor_1, length_scale_1)
            + cov_exp_quad(to_array_1d(x2[, m]), scale_factor_2, length_scale_2)
            + cov_exp_quad(to_array_1d(x3[, m]), scale_factor_3, length_scale_3)
            + cov_exp_quad(to_array_1d(x4[, m]), scale_factor_4, length_scale_4)
            + cov_exp_quad(to_array_1d(x5[, m]), scale_factor_5, length_scale_5);
        L = cholesky_decompose(K);
        exp_gp = exp(L * eta);
        for (n in 1:(N2)) {
            K[n, n] = K[n, n] + delta;
            Rt[n, m] *= exp_gp[n];
        }

        for (i in (N0 + 1):N2) {
            real convolution = y[m] * sum(SI[1:(i - 1)]);
            prediction[i, m] = Rt[i, m] * convolution;
        }
      
              E_deaths[1, m]= 1e-9;
      for (i in 2:N2){
        E_deaths[i,m]= 0;
        for(j in 1:(i-1)){
          E_deaths[i,m] += prediction[j,m]*f[i-j,m];
        }
      }
    }
    /* for(m in 1:M) {
        for(i in 1:N[m]) {
            LowerBound[i,m] = prediction[i,m] * 10 - cases[i,m];
        }
    }*/
}

model {
    tau_unit ~ exponential(1); // implies tau ~ exponential(0.03)
    target += -sum(y_raw); // exponential(1) prior on y_raw implies y ~ exponential(1 / tau)
    kappa_std ~ std_normal();
    phi_std ~ std_normal();
    mu ~ normal(2.4, kappa); // citation needed 
    alpha ~ gamma(0.5, 1);

    eta ~ std_normal();
    // GP priors
    scale_factor_1 ~ std_normal();
    length_scale_1 ~ std_normal();
    scale_factor_2 ~ std_normal();
    length_scale_2 ~ std_normal();
    scale_factor_3 ~ std_normal();
    length_scale_3 ~ std_normal();
    scale_factor_4 ~ std_normal();
    length_scale_4 ~ std_normal();
    scale_factor_5 ~ std_normal();
    length_scale_5 ~ std_normal();

    for(m in 1:M) {
        deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(
            E_deaths[EpidemicStart[m]:N[m], m], 
            phi
        );
    }
}

generated quantities {
    matrix[N2, M] lp0 = matrix_1000; // log-probability for LOO for the counterfactual model
    matrix[N2, M] lp1 = matrix_1000; // log-probability for LOO for the main model
    matrix[N2, M] prediction0 = matrix_0;
    matrix[N2, M] E_deaths0  = matrix_0;
    for (m in 1:M) {
        prediction0[1:N0, m] = rep_vector(y[m], N0); // learn the number of cases in the first N0 days
        for (i in (N0+1):N2) {
            real convolution0 = 0;
            for(j in 1:(i-1)) {
                convolution0 += prediction0[j, m] * SI[i-j]; // Correctd 22nd March
            }
            prediction0[i, m] = mu[m] * convolution0;
        } 
      
        E_deaths0[1, m]= 1e-9;
      for (i in 2:N2){
        E_deaths0[i,m]= 0;
        for(j in 1:(i-1)){
          E_deaths0[i,m] += prediction0[j,m]*f[i-j,m];
        }
      }

        for(i in 1:N[m]) {
            lp0[i, m] = neg_binomial_2_lpmf(deaths[i, m] | E_deaths[i, m], phi); 
            lp1[i, m] = neg_binomial_2_lpmf(deaths[i, m] | E_deaths0[i, m], phi); 
        }
    }
}
