"""
JobsData Column Definitions
==============================
Shared column name mapping for the headerless JobsData.csv (143 columns).
Validated by scripts/20_jobsdata_eda.py — all 8/8 position checks passed.
"""

JOBSDATA_COLUMNS = [
    # --- Core Job Info (0-7) ---
    "JobId",                    # 0
    "JobName",                  # 1
    "JobType",                  # 2
    "JobStatus",                # 3
    "StartDate",                # 4
    "EndDate",                  # 5
    "DueDate",                  # 6
    "NetFee",                   # 7

    # --- Job Classification (8-14) ---
    "JobPurpose",               # 8
    "Deliverable",              # 9
    "AppraisalFileType",        # 10
    "BusinessSegment",          # 11
    "BusinessSegmentDetail",    # 12
    "PortfolioMultiProperty",   # 13
    "PotentialLitigation",      # 14

    # --- Temporal Fields (15-27) ---
    "Year",                     # 15
    "WeekNumber",               # 16
    "DayOfWeek",                # 17
    "YearWeek",                 # 18
    "YearMonth",                # 19
    "YearQuarter",              # 20
    "WeekOfYear",               # 21
    "MonthNumber",              # 22
    "YearMonthNum",             # 23
    "YearWeekNum",              # 24
    "YearWeekConcat",           # 25
    "WeekNum2",                 # 26
    "MonthNum2",                # 27

    # --- Job Geography & Duration (28-31) ---
    "JobDistanceMiles",         # 28
    "OfficeJobTerritory",       # 29
    "JobLength_Days",           # 30
    "PropertyID",               # 31

    # --- Property Classification (32-41) ---
    "PropertyType",             # 32
    "SubType",                  # 33
    "SpecificUse",              # 34
    "GrossLandAreaAcres",       # 35
    "GrossLandAreaSF",          # 36
    "Shape",                    # 37
    "Topography",               # 38
    "BuildingClass",            # 39
    "YearBuilt",                # 40
    "PropertyCondition",        # 41

    # --- Property Location (42-47) ---
    "CityMunicipality",        # 42
    "StateName",                # 43
    "Market",                   # 44
    "Submarket",                # 45
    "CountyID",                 # 46
    "MarketOrientation",        # 47

    # --- Property Metrics (48-52) ---
    "PhotosCount",              # 48
    "SaleCount",                # 49
    "IECount",                  # 50
    "LeaseCount",               # 51
    "RentSurveyCount",          # 52

    # --- Geography (53-57) ---
    "IRRRegion",                # 53
    "RooftopLatitude",          # 54
    "RooftopLongitude",         # 55
    "GrossBuildingSF",          # 56
    "GLARentableSF",            # 57

    # --- Office Info (58-61) ---
    "OfficeID",                 # 58
    "OfficeCode",               # 59
    "CompanyLocation",          # 60
    "Office_Region",            # 61

    # --- Client Info (62-66) ---
    "ClientCompanyID",          # 62
    "CompanyName",              # 63
    "ClientContactID",          # 64
    "ContactType",              # 65
    "CompanyType",              # 66

    # --- ZipCodeMaster Demographics (67-142) — 76 columns ---
    "ZipCode",                  # 67
    "Zip_Latitude",             # 68
    "Zip_Longitude",            # 69
    "Zip_StateAbbr",            # 70
    "Zip_StateName",            # 71
    "Zip_DecommissionedFlag",   # 72
    "Zip_CityName",             # 73
    "Zip_CountyName",           # 74
    "Zip_CountyFIPS",           # 75
    "Zip_Congressional",        # 76
    "Zip_MSA",                  # 77
    "Zip_CBSA",                 # 78
    "Zip_CBSA2",                # 79
    "Zip_Congressional2",       # 80
    "Zip_Congressional3",       # 81
    "Zip_MultiCounty",          # 82
    "Zip_FIPS",                 # 83
    "Zip_CityType",             # 84
    "Zip_CityAliasAbbr",        # 85
    "Zip_PreferredFlag",        # 86
    "Zip_DecommFlag2",          # 87
    "Zip_CBSAName",             # 88
    "Zip_CBSAType",             # 89
    "Zip_Population",           # 90
    "Zip_HousingUnits",         # 91
    "Zip_CSAName",              # 92
    "Zip_CSACode",              # 93
    "Zip_CBSACode2",            # 94
    "Zip_Combined_Code",        # 95
    "Zip_CensusTract",          # 96
    "Zip_MSAName",              # 97
    "Zip_CMSAName",             # 98
    "Zip_PMSAName",             # 99
    "Zip_CensusRegion",         # 100
    "Zip_CensusDivision",       # 101
    "Zip_CensusDivisionCode",   # 102
    "Zip_CountyFIPS2",          # 103
    "Zip_PopulationEstimate",   # 104
    "Zip_HouseholdsPerZip",     # 105
    "Zip_AverageHouseValue",    # 106
    "Zip_IncomePerHousehold",   # 107
    "Zip_PersonsPerHousehold",  # 108
    "Zip_AverageHouseholdSize", # 109
    "Zip_MedianAge",            # 110
    "Zip_MedianAgeMale",        # 111
    "Zip_MedianAgeFemale",      # 112
    "Zip_DeliveryTotal",        # 113
    "Zip_SingleDelivery",       # 114
    "Zip_MultiDelivery",        # 115
    "Zip_GrowthRank",           # 116
    "Zip_GrowthIncrease",       # 117
    "Zip_MedianHouseValue",     # 118
    "Zip_AvgIncomePerHousehold",# 119
    "Zip_AvgHouseValue2",       # 120
    "Zip_MedianIncome",         # 121
    "Zip_NumberOfBusinesses",   # 122
    "Zip_NumberOfEmployees",    # 123
    "Zip_BusinessFirst8",       # 124
    "Zip_BusinessSecond8",      # 125
    "Zip_CompanyEmployees1k",   # 126
    "Zip_CompanyEmployees1kPct",# 127
    "Zip_CompanyEmployeesAll",  # 128
    "Zip_CompanyEmployeesAllPct",# 129
    "Zip_WorkersInZip",         # 130
    "Zip_WorkersOutZip",        # 131
    "Zip_WhiteCollar",          # 132
    "Zip_BlueCollar",           # 133
    "Zip_DeliveryResidential",  # 134
    "Zip_DeliveryBusiness",     # 135
    "Zip_DeliveryOther",        # 136
    "Zip_DeliveryAll",          # 137
    "Zip_LandArea",             # 138
    "Zip_WaterArea",            # 139
    "Zip_TotalDensity",         # 140
    "Zip_PopDensity",           # 141
    "Zip_PopCount",             # 142
]

assert len(JOBSDATA_COLUMNS) == 143, f"Expected 143 columns, got {len(JOBSDATA_COLUMNS)}"
