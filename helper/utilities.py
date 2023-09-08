import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class Mixen:
    """
    Class for mixing and processing audio segments.
    """

    def get_segments(self, dfx, col):
        """
        Get the groups with the highest score based on a threshold.

        Args:
            dfx (pandas.DataFrame): Dataframe containing the scores.
            col (str): The column name containing the scores.

        Returns:
            list: List of groups with high scores.
        """
        thresh = 0.65 * (max(dfx[col]) / 2)
        segments = []
        segment = []
        for index, row in dfx[dfx[col] > thresh].iterrows():
            if len(segment) == 0:
                segment.append(index)

            if index - segment[-1] < 4:
                segment.append(index)
            else:
                segment.sort()
                continus_segment = list(range(segment[0], segment[-1] + 1))
                segments.append(continus_segment)
                segment = []
        segments = sorted(segments, key=len, reverse=True)
        return segments

    def subtract_time(self, val2, val1):
        """
        Subtract two time values in the format HH:MM:SS.

        Args:
            val2 (str): Time value to subtract.
            val1 (str): Time value to subtract from.

        Returns:
            str: Subtraction result in HH:MM:SS format.
        """
        val1 = list(map(float, val1.split(":")))
        val2 = list(map(float, val2.split(":")))

        if val2[-1] < val1[-1]:
            val2[-1] += 60
            val2[-2] -= 1

        if val2[-2] < val1[-2]:
            val2[-2] += 60
            val2[-3] -= 1

        return ":".join([str(round(val2[i] - val1[i], 3)) for i in range(len(val1))])

    def calc_time(self, segment, dfx):
        """
        Calculate the total duration time for a group.

        Args:
            segment (list): List of indices representing the group.
            dfx (pandas.DataFrame): Dataframe containing the segments.

        Returns:
            float: Total duration time in seconds.
        """
        start = segment[0]
        end = segment[-1]

        Total_Duration = self.subtract_time(dfx.iloc[end]["Out"], dfx.iloc[start]["In"])
        time_list = list(map(float, Total_Duration.split(":")))
        time = time_list[0] * 60 * 60 + time_list[1] * 60 + time_list[2]
        return time

    def get_60s(self, segmentx, dfx):
        """
        Divide a group into subgroups of less than 60 seconds.

        Args:
            segmentx (list): List of indices representing the group.
            dfx (pandas.DataFrame): Dataframe containing the segments.

        Returns:
            list: List of subgroups with duration less than 60 seconds.
        """
        new_segment = []

        if len(segmentx) <= 1:
            new_segment.append(segmentx)
            return new_segment

        for j in range(len(segmentx)):
            time = self.calc_time(segmentx[0 : j + 1], dfx)

            if time > 60 and len(segmentx[0 : j + 1]) > 1:
                seg1 = segmentx[0:j]
                seg2 = segmentx[j:]
                new_segment.append(seg1)
                # recursion
                self.get_60s(seg2, dfx)
                return new_segment

    def devide_long_segments_into_60s(self, segments, dfx):
        """
        Divide long segments into subsegments of less than 60 seconds.

        Args:
            segments (list): List of indices representing the segments.
            dfx (pandas.DataFrame): Dataframe containing the segments.

        Returns:
            list: List of subsegments with duration less than 60 seconds.
        """
        new_segments = []
        for i, segment in enumerate(segments):
            if self.calc_time(segment, dfx) <= 60 and self.calc_time(segment, dfx) >= 3:
                new_segments.append(segment)
            elif self.calc_time(segment, dfx) > 60:
                new_segment = []
                new_segment = self.get_60s(segment, dfx)

                for i, seg in enumerate(new_segment):
                    if self.calc_time(seg, dfx) < 3:
                        del new_segment[i]
                        continue

                new_segments.extend(new_segment)
        return new_segments

    def Concat_time(self, dfx):
        """
        Add a column for the total duration time to the dataframecontaining the segments.

        Args:
            dfx (pandas.DataFrame): Dataframe containing the segments.

        Returns:
            pandas.DataFrame: Dataframe with the total duration time column added.
        """
        dfx['Total Duration'] = self.subtract_time(dfx.iloc[-1]["Out"], dfx.iloc[0]["In"])
        return dfx

    def process_df(self, dfx, new_segments):
        """
        Process the dataframe based on the groups and subgroups.

        Args:
            dfx (pandas.DataFrame): Dataframe containing the segments.
            new_segments (list): List of subsegments with duration less than 60 seconds.

        Returns:
            pandas.DataFrame: Processed dataframe with groups and total duration time.
        """
        df_merged = dfx.iloc[new_segments[0][0] : new_segments[0][-1] + 1]
        df_merged['group'] = '1'
        df_merged = self.Concat_time(df_merged)

        for i, segment in enumerate(new_segments):
            if i == 0:
                continue
            df1 = dfx.iloc[segment[0] : segment[-1] + 1]

            df1['group'] = str(i + 1)
            df1 = self.Concat_time(df1)
            df_merged = pd.concat([df_merged, df1], axis=0)
        return df_merged.set_index(['Total Duration', 'group'])