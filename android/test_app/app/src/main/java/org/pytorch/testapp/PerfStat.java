package org.pytorch.testapp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PerfStat {

  private final List<Long> mList = new ArrayList<>();
  private long mSum = 0;
  public PerfStat() {
  }

  public void add(long duration) {
    mList.add(duration);
    mSum += duration;
  }

  private int percentileIdx(int n, int percent) {
    return (int) Math.ceil((n - 1) * percent / 100.f);
  }

  private final int[] STAT_STRING_PERCENTILES = new int[] {10, 25, 50, 75, 90};

  public String getStatString() {
    Collections.sort(mList);
    int n = mList.size();
    float avg = (float) mSum / n;
    long min = mList.get(0);
    long max = mList.get(n - 1);

    StringBuffer sb = new StringBuffer();
    sb.append("N:").append(n)
        .append(" avg:").append(avg)
        .append(" min:").append(min)
        .append(" max:").append(max);
    for (int percent : STAT_STRING_PERCENTILES) {
      long percentile = mList.get(percentileIdx(n, percent));
      sb.append(" p").append(percent).append(":").append(percentile);
    }
    return sb.toString();
  }
}
