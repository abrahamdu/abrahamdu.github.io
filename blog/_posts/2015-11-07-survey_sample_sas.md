---
layout: post
section-type: post
published: true
category: SAS
tags: [ 'sas' ]
title: Survey Sampling for Quality Control
date: 2015-11-07
modifiedOn: 2015-12-23
---

This is my first SAS paper I prepared for [2014 SAS Global Forum](http://support.sas.com/events/sasglobalforum/2014/) and presented on March 23, 2014. The idea comes from a JSM working paper which uses box-and-whisker outliers to re-stratify the total population for sampling. The final sample size is still determined by optimal allocation which takes account of the size of each stratum, the expected variance and survey cost. I implemented my idea by applying SAS and people interested in my idea are welcome to use the SAS macro shared in the appendix of [this paper](http://support.sas.com/resources/papers/proceedings14/1630-2014.pdf). 

<script async class="speakerdeck-embed" data-id="e17f46361fa94675bd7f121a700f379b" data-ratio="1.77777777777778" src="//speakerdeck.com/assets/embed.js"></script>

Appendix:  

```sql
%MACRO
STRA_SAMP_OUTLIER(POP,STRATA,K,ONE_SIDED,VAR,COST,TOT_SAMP,ALLOCMETH,A
LLOCMIN,OUT_POP);
/*****************************************************************/
/* This customized SAS macro is to implement modified stratified */
/* sampling framework by re-defining strata from box-and-whisker */
/* outliers by Yi Du 2014 SAS Global Forum Paper. */
/* Author: Yi Du */
/* Macro Variable Definition: */
/* &POP.: the whole population used for sampling */
/* &STRATA.: the variable(s) to define stratum */
/* &K.: positive number used to calculate outliers */
/* &ONE_SIDED.: Y/N to indicate to group left-sided outliers or */
/* or both-sided outliers */
/* &VAR.: the variable(s) used to measure variance by strata */
/* &COST.: the average value of variable(s) to measure cost */
/* &TOT_SAMP.: the total sample size requested */
/* &ALLOCMETH.: Optimal, Neyman or proportional allocation method*/
/* &ALLOCMIN.: the minimum number of units sampled by strata */
/* &OUT_POP.: the output SAS data set with sample information */
/*****************************************************************/
%LET i = 1;
 %DO %WHILE(%LENGTH(%SCAN(&strata.,%EVAL(&i.))) > 0);
 %IF %EVAL(&i.) = 1 %THEN %DO;
 %LET newstrata = %SCAN(&strata.,&i.);
%END;
%ELSE %DO;
 %LET newstrata = &newstrata.%STR(*)%SCAN(&strata.,%EVAL(&i.));
%END;
%LET i = %EVAL(&i.+1);
 %END;
Proc Freq Data = &pop.;
 Tables &newstrata. / Noprint Out = freq_&pop.;
Run;
Proc Univariate Data = freq_&pop.;
 Var count;
 Output Out = quartile_&pop. Q1 = q1 Q3 = q3 Qrange = IQR Min = min
Max = max;
Run;
Proc Sql Noprint;
 Select min,q1,q3,iqr,max Into :min,:q1,:q3,:iqr,:max
 From quartile_&pop.;
Quit;
%IF &min. >= %SYSEVALF(&q1. - &k.*&iqr) %THEN %DO;
 %PUT ERROR: No outliers identified in the population. Please
consider other survey sample methods.;
 %GOTO exit;
%END;
%ELSE %DO;
Proc Sort Data = &pop. Out = sorted_&pop.;
 By &strata.;
Proc Sort Data = freq_&pop.;
 By &strata.;
Run;
Data new_&pop.;
 Merge sorted_&pop. (in=froms) freq_&pop. (in=fromc);
By &strata.;
If froms and fromc Then Output;
Run;
Data restra_&pop.;
 Set new_&pop.;
%LET j = 1;
%LET new_strata =;
%DO %WHILE(%LENGTH(%SCAN(&strata.,%EVAL(&j.))) > 0);
 new_%SCAN(&strata.,&j.) = %SCAN(&strata.,&j.);
 %LET new_strata = &new_strata.%NRSTR( )new_%SCAN(&strata.,&j.);
 %LET j = %EVAL(&j.+1);
%END;
%PUT &new_strata.;
%IF &one_sided. = N %THEN %DO;
 If count < %SYSEVALF(&Q1.- &k.*&IQR.) Or count > %SYSEVALF(&Q3.
+ &k.*&IQR.) Then Do;
 %DO k = 1 %TO %EVAL(&j.-1);
 Format %SCAN(&new_strata.,&k.) $32.;
 %SCAN(&new_strata.,&k.) = 'Combined';
%END;
 End;
 Else Do;
 %DO k = 1 %TO %EVAL(&j.-1);
 Format %SCAN(&new_strata.,&k.) $32.;
 %SCAN(&new_strata.,&k.) = %SCAN(&strata.,&k.);
%END;
 End;
%END;
%ELSE %DO;
 If count < %SYSEVALF(&Q1.- &k.*&IQR.) Then Do;
 %DO k = 1 %TO %EVAL(&j.-1);
 Format %SCAN(&new_strata.,&k.) $32.;
 %SCAN(&new_strata.,&k.) = 'Combined';
%END;
 End;
 Else Do;
 %DO k = 1 %TO %EVAL(&j.-1);
 Format %SCAN(&new_strata.,&k.) $32.;
 %SCAN(&new_strata.,&k.) = %SCAN(&strata.,&k.);
%END;
 End;
%END;
Run;
%IF &var. ^= %THEN %DO;
Proc Univariate Data = restra_&pop. Noprint;
 Var &var.;
 Class &new_strata.;
 Output Out = var Var=_var_;
Run;
%END;
%IF &cost. ^= %THEN %DO;
Proc Univariate Data = restra_&pop. Noprint;
 Var &cost.;
 Class &new_strata.;
 Output Out = cost Mean = _cost_;
Run;
Proc Sort Data = var;
 By &new_strata.;
Proc Sort Data = cost;
 By &new_strata.;
Run;

Data var;
 Merge var (in=fromv) cost (in=fromc);
 By &new_strata.;
If fromv and fromc Then Output;
Run;
%END;
Proc Sort Data = restra_&pop.;
 By &new_strata.;
Run;
Proc Surveyselect Data = restra_&pop. Seed = 9234 Method = srs Out =
&out_pop. Sampsize = &tot_samp.;
 Stratum &new_strata./
 %IF %UPCASE(&allocmeth.) = OPTIMAL %THEN %DO;
 alloc = &allocmeth. var = var cost = cost allocmin = &allocmin.;
 %END;
 %ELSE %IF %UPCASE(&allocmeth.) = NEYMAN %THEN %DO;
 alloc = &allocmeth. var = var allocmin = &allocmin.;
 %END;
 %ELSE %DO;
 alloc = &allocmeth. allocmin = &allocmin.;
 %END;
Run;
%END;
%EXIT:
%MEND STRA_SAMP_OUTLIER;
```