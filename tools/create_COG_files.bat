set SITE=<SITE_NAME>&
set PROCESS_DIR=<PRODUCT_DIR>&
gdal_translate -of COG -tr 0.1 0.1 -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE %PROCESS_DIR%\%SITE%\Ortho.vrt %SITE%_Ortho.tif &
gdal_translate -of COG -tr 0.1 0.1 -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE %PROCESS_DIR%\%SITE%\DSM.vrt %SITE%_DSM.tif &
gdaldem hillshade -multidirectional -of COG -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE %SITE%_DSM.tif %SITE%_DSM_hillshade.tif