@echo off
echo "Start remove dirt"
for /D %%f in (catboost_1*.*) do rmdir "%%f" /Q /S
rmdir .tmp.driveupload /Q /S
rmdir .tmp.drivedownload /Q /S
rmdir catboost_info /Q /S

echo "Done remove dirt"