########## Utility functions for scripts ##########

"""
    load_data(infile::String, pm::NamedTuple)

Load a time series from `infile` with the parameter set `pm`. Example of `pm`:
```
parms = (
    # for intraday data only
    intraday = true,
    # date range of data
    date_start = Dates.DateTime("0000-01-01"), date_end = Dates.DateTime("9999-01-01"),
    # daily time range
    wa = Dates.Hour(9) + Dates.Minute(30), wb = Dates.Hour(16) + Dates.Minute(0),
    adjusted = false,  # adjust intraday data

    # index of the column in the dataframe to work on
    column_index = 1,
)
```
"""
function load_data(infile::String, pm::NamedTuple)

    toto0 = TimeSeries.readtimearray(infile, format="yyyy-mm-dd HH:MM:SS", delim=',')
    cname = TimeSeries.colnames(toto0)[pm.column_index]
    toto0 = toto0[cname]

    if pm.intraday  # load intraday data
        toto = TimeSeries.to(TimeSeries.from(toto0, pm.date_start), pm.date_end)

        # Data splitted per day
        DataSplt = FracFin.window_split_timearray(toto, Dates.Day(1), (pm.wa, pm.wb); fillmode=:fb)
        #   Npd = size(DataSplt[1])[1]  # number of samples per day

        Data = if pm.adjusted
            vcat(FracFin.equalize_daynight(DataSplt)...)  # data with overnight effect removed
        else
            vcat(DataSplt...)  # raw data
        end

        return (DataRaw=Data, DataSplt=DataSplt)
    else
        return (DataRaw=toto0, DataSplt=nothing)
    end
end
