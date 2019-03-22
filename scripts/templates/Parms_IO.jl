########## Parameters for scripts ##########

######## parameters for loading data ########
parms_io = (
    intraday = true,  # true for intraday data
    # date range of data
    date_start = Dates.DateTime("0000-01-01"), date_end = Dates.DateTime("9999-01-01"),
    # daily time range
    wa = Dates.Hour(9) + Dates.Minute(30), wb = Dates.Hour(15) + Dates.Minute(59),
    adjusted = false,  # adjust intraday data

    # index of the column in the dataframe to work on
    column_index = 1,
)

