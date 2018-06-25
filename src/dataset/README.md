### Data Preparing
Main scripts for building the training dataset.

#### [*Spider/getchu_get_urls.py*](./Spider/getchu_get_urls.py)
Parse all urls from *erogamescape_sql*''s result file.
You can get the *erogamescape_sql* file using [/scripts/get_sql_results.sh](../../scripts/get_sql_results.sh)

#### [*Spider/getchu_get_raw_image.py*](./Spider/getchu_get_raw_image.py)
Download all the images and messages from getchu.com.

#### [*Spider/streamline_rough.py*](./Spider/streamline_rough.py)
Cleaning the images roughly. Check this python file for more details.

#### [*Spider/utils.py*](./Spider/utils.py)
Some useful functions.

#### [*FaceDetect/detector.py*](./FaceDetect/detector.py)
For anime face detecting. Save the images to a specific path.

#### [*FaceDetect/streamline.py*](./FaceDetect/streamline.py)
Cleaning the images roughly. Check this python file for more details.

#### [*FaceDetect/utils.py*](./FaceDetect/utils.py)
Some useful functions.
