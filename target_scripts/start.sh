sudo ~/membw_ctrl --platform imx8m init > /dev/null
if [ -z "$1" ] ; then
	BW=0
else
	BW=$1
fi

if [ -z "$2" ] ; then
	INF=0
else
	INF=$2
fi

BW=$(expr $BW \* 2)
sudo ~/membw_ctrl --platform imx8m start $BW $INF $INF 0 0 > /dev/null

