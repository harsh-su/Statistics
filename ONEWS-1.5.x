Summary of important user-visible changes for statistics 1.5.0:
-------------------------------------------------------------------

 Important Notice: 1) dependency change to Octave>=6.1.0
                   2) `mean` shadows core Octave's respective function
                   3) removed dependency on `io` package
                   4) incompatibility with the `nan` package
 New functions:
 ==============

 ** anova2 (fully Matlab compatible)

 ** bvncdf

 ** cdfcalc, cdfplot

 ** chi2gof (fully Matlab compatible. bug #46764)

 ** chi2test (bug #58838)

 ** cholcov (fully Matlab compatible)

 ** ecdf (fully Matlab compatible)

 ** evfit (fully Matlab compatible)

 ** evlike (fully Matlab compatible)

 ** fitlm (mostly Matlab compatible)

 ** fillmissing (patch #10102)

 ** friedman (fully Matlab compatible)

 ** grpstats (complementary to manova1)

 ** kruskalwallis (fully Matlab compatible)

 ** kstest (fully Matlab compatible)

 ** kstest2 (fully Matlab compatible. bug #56572)

 ** libsvmread, libsvmwrite (I/O functions for LIBSVM data files)

 ** manova1 (fully Matlab compatible)

 ** manovacluster (fully Matlab compatible)

 ** mean (fully Matlab compatible, it shadows mean from core Octave)

 ** multcompare (fully Matlab compatible)

 ** mvtcdfqmc

 ** ranksum (fully Matlab compatible. bug #42079)

 ** standardizeMissing (patch #10102)

 ** svmpredict, svmtrain (wrappers for LIBSVM 3.25)

 ** tiedrank (complementary to ranksum)

 ** x2fx (missing function: bug #48146)

 Improvements:
 =============

 ** anova1: added extra feature for performing Welch's ANOVA (PR #15)

 ** anovan: mostly Matlab compatible, extra features. (patch #10123, PR #1-42)

 ** binopdf: implement high accuracy Loader algorithm for m>=10 (bug #34362)

 ** cdf: extended to include all available distributions

 ** crosstab: can handle char arrays, fixed ordering of groups

 ** gaminv: fixed accuracy for small 1st argument (bug #56453)

 ** geomean: fully Matlab compatible. (patch #59410)

 ** gevlike: fully Matlab compatible. ACOV output is the inverse Fisher Inf Mat

 ** grp2idx (fully Matlab compatible, indexes in order of appeearance)

 ** harmmean: fully Matlab compatible.

 ** hygepdf: added optional parameter "vectorexpand" to facilitate
    vectorization of other hyge functions. Allows different inputs lengths
    for x and t,m,n parameters, with broadcast expanded output (bug #34363)

 ** hygecdf: improved vectorization for non-scalar inputs.
    hygeinv
    hygernd

 ** ismissing: corrects handling of n-D arrays, NaN indicators, and improves
    matlab compatibility for different data types. (patch #10102)

 ** kmeans: improved help file, evaluate efficiency (bug #8959)

 ** laplace_cdf: allow for parameters mu and scale (bug #58688)
    laplace_inv
    laplace_pdf
    logistic_cdf
    logistic_inv
    logistic_pdf

 ** logistic_regression: fixed incorrect results (bug #60348)

 ** mvncdf: improved performance and accuracy (bug #44130)

 ** normplot: fixed ploting error (bug #62394), updated features

 ** pdf: extended to include all available distributions

 ** pdist: updated the 'cosine' metric to be more efficient (bug #62495)

 ** rmmissing: corrects cellstr array handling and
    improves matlab compatibility for different data types. (patch #10102)

 ** signtest: fix erroneous results, fully Matlab compatible (bug #49961)

 ** ttest2: can handle NaN values as missing data (bug #58697)

 ** violin: fix parsing color vector affecting Octave>=6.1.0 (bug #62805)

 ** wblplot: fixed coding style and help texinfo. (patch #8579)

 Removed Functions:
 ==================

 ** anova (replaced by anova1)

 ** caseread, casewrite (do not belong here)

 ** chisquare_test_homogeneity (replaced by chi2test)

 ** chisquare_test_independence (replaced by chi2test)

 ** kolmogorov_smirnov_test (replaced by kstest)

 ** kolmogorov_smirnov_test_2 (replaced by kstest2)

 ** kruskal_wallis_test (replaced by kruskalwallis)

 ** manova (replaced by manova1)

 ** repanova (replaced by anova2)

 ** sign_test (replaced by updated signtest)

 ** tblread, tblwrite (belong to `io` package when tables are implemented)

 ** t_test, t_test_2 (deprecated: use ttest & ttest2)

 ** wilcoxon_test (replaced by ranksum)

 Available Data Sets:
 ====================

 ** acetylene         Chemical reaction data with correlated predictors
 ** arrhythmia        Cardiac arrhythmia data from the UCI machine learning repository
 ** carbig            Measurements of cars, 1970–1982
 ** carsmall          Subset of carbig. Measurements of cars, 1970, 1976, 1982
 ** cereal            Breakfast cereal ingredients
 ** examgrades        Exam grades on a scale of 0–100
 ** fisheriris        Fisher's 1936 iris data
 ** hald              Heat of cement vs. mix of ingredients
 ** heart_scale.dat   Used for SVM testing
 ** kmeansdata        Four-dimensional clustered data
 ** mileage           Mileage data for three car models from two factories
 ** morse             Recognition of Morse code distinctions by non-coders
 ** popcorn           Popcorn yield by popper type and brand
 ** stockreturns      Simulated stock returns
 ** weather           Daily high temperatures in the same month in two consecutive years


Summary of important user-visible changes for statistics 1.5.1:
-------------------------------------------------------------------

 Important Notice: 1) `mean` shadows core Octave's respective function
                   2) incompatibility with the `nan` package
 New functions:
 ==============

 ** barttest (fully Matlab compatible)

 ** evcdf (fully Matlab compatible)

 ** evinv (fully Matlab compatible)

 ** evpdf (fully Matlab compatible)

 ** evrnd (fully Matlab compatible)

 ** evstat (fully Matlab compatible)

 ** gpfit (fully Matlab compatible)

 ** gplike (fully Matlab compatible)

 ** gpstat (fully Matlab compatible)

 ** levene_test (options for testtypes, handling NaNs and GROUPS like anova1)

 ** ncfcdf (fully Matlab compatible)

 ** ncfinv (fully Matlab compatible)

 ** ncfpdf (fully Matlab compatible)

 ** ncfrnd (fully Matlab compatible)

 ** ncfstat (fully Matlab compatible)

 ** nctcdf (fully Matlab compatible)

 ** nctinv (fully Matlab compatible)

 ** nctpdf (fully Matlab compatible)

 ** nctrnd (fully Matlab compatible)

 ** nctstat (fully Matlab compatible)

 ** ncx2cdf (fully Matlab compatible)

 ** ncx2inv (fully Matlab compatible)

 ** ncx2rnd (fully Matlab compatible)

 ** ncx2stat (fully Matlab compatible)

 ** normlike (fully Matlab compatible)

 ** sampsizepwr (fully Matlab compatible with extra functionality)

 ** vartestn (fully Matlab compatible)

 Improvements:
 =============

 ** bartlett_test: improved functionality, hanlding NaNs and GROUPS like anova1

 ** chi2cdf: added "upper" option and confidence bounds

 ** chi2test: improved functionality, handles multi-way tables

 ** crosstab: returns chi-square and p-value for multiway tables

 ** evfit: fixed bug that caused an error when x is a row vector

 ** fcdf: added "upper" option and confidence bounds

 ** gamcdf: added "upper" option and confidence bounds

 ** gpcdf: added "upper" option and confidence bounds

 ** mvnpdf: fixed MATLAB compatibility

 ** mvnrnd: fixed MATLAB compatibility

 ** ncx2pdf: reimplemented to be fully MATLAB compatible

 ** normcdf: added "upper" option and confidence bounds

 ** tcdf: added "upper" option

 ** vartest: fixed MATLAB compatibility

 ** vartest2: fixed MATLAB compatibility

 ** ztest: fixed MATLAB compatibility

 Removed Functions:
 ==================

 ** cloglog

 ** nanmean (replaced by mean)

 ** kolmogorov_smirnov_cdf (unused by new kstest, kstest2 functions)

 ** u_test (replaced by ranksum)

 ** var_test (replaced by vartest)

 ** z_test (replaced by ztest)


 Summary of important user-visible changes for statistics 1.5.2:
-------------------------------------------------------------------

 Important Notice: 1) `mean`, `median`, `std`, and `var` functions
                       shadow core Octave's respective functions
                   2) incompatibility with the `nan` package
 New functions:
 ==============

 ** median (fully Matlab compatible)

 ** std (fully Matlab compatible)

 ** var (fully Matlab compatible)

 Improvements:
 =============

 ** mean: fixed MATLAB compatibility

 ** multcompare: fixed erroneous results for Welch ANOVA, updated features

 ** tcdf: fixed erroneous results

 ** ttest: added support for NaN values and matrix inputs

 ** ttest2: added support for matrices and multiple t-tests

 Removed Functions:
 ==================

 ** nanmedian (replaced by median)

 ** nanstd (replaced by std)

 ** nanvar (replaced by var)


 Summary of important user-visible changes for statistics 1.5.3:
-------------------------------------------------------------------

 Important Notice: 1) `mean`, `median`, `std`, and `var` functions
                       shadow core Octave's respective functions
                   2) incompatibility with the `nan` package
 New functions:
 ==============

 ** adtest (fully Matlab compatible)

 ** hotelling_t2test (new functionality, replacing old hotelling_test)

 ** hotelling_t2test2 (new functionality, replacing old hotelling_test_2)

 ** regression_ftest (new functionality, replacing old f_test_regression)

 ** regression_ttest (replacing old t_test_regression)

 ** vmcdf (von Mises cummulative distribution function)

 Improvements:
 =============

 ** betacdf: added "upper" option

 ** binocdf: added "upper" option

 ** expcdf: added "upper" option and confidence bounds

 ** geocdf: added "upper" option

 ** gevcdf: added "upper" option

 ** hygecdf: added "upper" option

 ** laplace_cdf: updated functionality

 ** laplace_inv: updated functionality

 ** laplace_pdf: updated functionality

 ** laplace_rnd: updated functionality

 ** logistic_cdf: updated functionality

 ** logistic_inv: updated functionality

 ** logistic_pdf: updated functionality

 ** logistic_rnd: updated functionality

 ** logncdf: added "upper" option and confidence bounds

 ** mean: fixed MATLAB compatibility

 ** median: fixed MATLAB compatibility

 ** multcompare: print PostHoc Test table

 ** nbincdf: added "upper" option

 ** poisscdf: added "upper" option

 ** raylcdf: added "upper" option

 ** std: fixed MATLAB compatibility

 ** unidcdf: added "upper" option

 ** unifcdf: added "upper" option

 ** var: fixed MATLAB compatibility

 ** vmpdf: updated functionality

 ** vmrnd: updated functionality

 ** wblcdf: added "upper" option and confidence bounds

 Removed Functions:
 ==================

 ** anderson_darling_cdf (replaced by adtest)

 ** anderson_darling_test (replaced by adtest)

 ** hotelling_test (replaced by hotelling_t2test)

 ** hotelling_test_2 (replaced by hotelling_t2test2)

 ** f_test_regression (replaced by regression_ftest)

 ** t_test_regression (replaced by regression_ttest)


 Summary of important user-visible changes for statistics 1.5.4:
-------------------------------------------------------------------

 Important Notice: 1) `mean`, `median`, `std`, and `var` functions
                       shadow core Octave's respective functions
                   2) incompatibility with the `nan` package
 New functions:
 ==============

 ** bvtcdf

 ** correlation_test (new functionality, replacing old cor_test)

 ** icdf (wrapper for all available *inv distribution functions)

 ** fishertest (fully Matlab compatible)

 ** procrustes (fully Matlab compatible)

 ** ztest2 (new functionality, replacing old prop_test_2)

 Improvements:
 =============

 ** cdf: updated wrapper for all available *cdf distribution functions

 ** dcov: handles missing values and multivariate samples

 ** geomean: fixed MATLAB compatibility

 ** harmmean: fixed MATLAB compatibility

 ** mean: fixed MATLAB compatibility

 ** median: fixed MATLAB compatibility

 ** mvtcdf: improved speed, fixed Matlab compatibility

 ** pdf: updated wrapper for all available *pdf distribution functions

 ** random: updated wrapper for all available *rnd distribution functions

 ** regression_ttest: new functionality

 ** std: fixed MATLAB compatibility

 ** var: fixed MATLAB compatibility

 Removed Functions:
 ==================

 ** cor_test (replaced by correlation_test)

 ** prop_test_2 (replaced by ztest2)
