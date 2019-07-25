# Identify Fraud From Enron Email
Enron	 Corporation	 was	 an	 American	 energy	 company	 based	 in	 Houston,	
Texas,	formed	in	1985	by	Kenneth	Lay.	According	to	(https://en.wikipedia.org/wiki/Enron_scandal),
when Enron’s	CEO Jeffrey	Skilling [“was	hired,	he developed	a	staff	of	executives	that	- by	the	use	of	accounting	
loopholes,	special	purpose	entities,	and	poor	financial	reporting	- were	able	to	hide	
billions	 of	 dollars	 in	 debt	 from	 failed	 deals	 and	 projects.	 Chief	 Financial	 Officer	
Andrew	Fastow and	other	executives	not	only	misled	Enron's	Board	of	Directors	
and	Audit	Committee	on	high-risk	accounting	practices,	but	also	pressured	Arthur	
Andersen	to	ignore	the	issues”].

## Motivation
The motivation of this project is the ability of working with real world "messie" data,
which involves performing deep analysis and cleaning process,
as well as the ability to implement supervised learning techniques.

## Files in this Repository

**README.md:** the present file

**Final_Report.pdf:**	a detailed report of the project

**poi_id.py**: the python code with all the analyses

**tester.py:** a tester function that performs model validation

## Results
Data was cleaned, missing values were handled and outliers were removed.
The following keys ('people') were considered outliers in the dataset:

- WODRASKA	JOHN
- WHALEY	DAVID	A
- CLINE	KENNETH	W,	WROBEL	
- BRUCE
- SCRIMSHAW	MATTHEW
- GILLIS	JOHN
- THE	TRAVEL	AGENCY	IN	THE	PARK
- TOTAL

Two features were engineered (fraction_to_poi and	fraction_from_poi).

An exhaustive feature selection was performed together with SelectKBest, trying several
feature combinations.

The best obtained classifier was a **Naive Bayes** with the following results (obtained through
a stratifiedkfold)

- Accuracy: 0.85793.	That	means	85,793%	of	all	points	were	correctly	
classified	as POI (people	of	interest)	or Non-POI.
- Recall: 0.39550.	That	means	amongst	POI	and	Non-POI	in	the	testing	
set,	39,550%	of	the	POI	were	identified.
- Precision: 0.50350.	That	means	that	taking	all	the	classified	points,	
50,350%	of	them	were	correctly	classified	as	POI.
