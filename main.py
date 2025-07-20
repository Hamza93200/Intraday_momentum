Sub CalculateCurrencyMaturityAnalysis()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("Spot")
    
    ' Find the last row with data
    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    
    ' Assuming data starts from row 2 (row 1 has headers)
    Dim dataStartRow As Long
    dataStartRow = 2
    
    ' Define column positions
    Dim dateCol As Long, curr1Col As Long, curr2Col As Long
    dateCol = 1 ' Column A - DATES
    curr1Col = 2 ' Column B - GBPUSD Curncy
    curr2Col = 3 ' Column C - GBPHKD Curncy
    
    ' Find next available column for output
    Dim outputStartCol As Long
    outputStartCol = 4 ' Start from column D
    
    ' Create headers
    Call CreateHeaders(ws, outputStartCol)
    
    ' Calculate for each maturity
    Call CalculateDaily(ws, dataStartRow, lastRow, dateCol, curr1Col, curr2Col, outputStartCol)
    Call CalculateWeekly(ws, dataStartRow, lastRow, dateCol, curr1Col, curr2Col, outputStartCol + 4)
    Call CalculateMonthly(ws, dataStartRow, lastRow, dateCol, curr1Col, curr2Col, outputStartCol + 8)
    Call CalculateQuarterly(ws, dataStartRow, lastRow, dateCol, curr1Col, curr2Col, outputStartCol + 12)
    
    MsgBox "Currency maturity analysis completed!"
End Sub

Sub CreateHeaders(ws As Worksheet, startCol As Long)
    ' Daily headers
    ws.Cells(1, startCol).Value = "Daily_Abs_Diff_Curr1"
    ws.Cells(1, startCol + 1).Value = "Daily_Abs_Diff_Curr2"
    ws.Cells(1, startCol + 2).Value = "Daily_Rel_Change_Curr1"
    ws.Cells(1, startCol + 3).Value = "Daily_Rel_Change_Curr2"
    
    ' Weekly headers
    ws.Cells(1, startCol + 4).Value = "Weekly_Abs_Diff_Curr1"
    ws.Cells(1, startCol + 5).Value = "Weekly_Abs_Diff_Curr2"
    ws.Cells(1, startCol + 6).Value = "Weekly_Rel_Change_Curr1"
    ws.Cells(1, startCol + 7).Value = "Weekly_Rel_Change_Curr2"
    
    ' Monthly headers
    ws.Cells(1, startCol + 8).Value = "Monthly_Abs_Diff_Curr1"
    ws.Cells(1, startCol + 9).Value = "Monthly_Abs_Diff_Curr2"
    ws.Cells(1, startCol + 10).Value = "Monthly_Rel_Change_Curr1"
    ws.Cells(1, startCol + 11).Value = "Monthly_Rel_Change_Curr2"
    
    ' Quarterly headers
    ws.Cells(1, startCol + 12).Value = "Quarterly_Abs_Diff_Curr1"
    ws.Cells(1, startCol + 13).Value = "Quarterly_Abs_Diff_Curr2"
    ws.Cells(1, startCol + 14).Value = "Quarterly_Rel_Change_Curr1"
    ws.Cells(1, startCol + 15).Value = "Quarterly_Rel_Change_Curr2"
End Sub

Sub CalculateDaily(ws As Worksheet, startRow As Long, lastRow As Long, dateCol As Long, curr1Col As Long, curr2Col As Long, outputCol As Long)
    Dim i As Long
    
    For i = startRow + 1 To lastRow ' Start from second data row
        If i <= lastRow And ws.Cells(i - 1, curr1Col).Value <> "" And ws.Cells(i, curr1Col).Value <> "" Then
            ' Calculate absolute differences
            Dim absDiff1 As Double, absDiff2 As Double
            absDiff1 = Abs(ws.Cells(i, curr1Col).Value - ws.Cells(i - 1, curr1Col).Value)
            absDiff2 = Abs(ws.Cells(i, curr2Col).Value - ws.Cells(i - 1, curr2Col).Value)
            
            ws.Cells(i, outputCol).Value = absDiff1
            ws.Cells(i, outputCol + 1).Value = absDiff2
            
            ' Calculate relative changes (absolute difference / initial value)
            If ws.Cells(i - 1, curr1Col).Value <> 0 Then
                ws.Cells(i, outputCol + 2).Value = absDiff1 / ws.Cells(i - 1, curr1Col).Value
            End If
            If ws.Cells(i - 1, curr2Col).Value <> 0 Then
                ws.Cells(i, outputCol + 3).Value = absDiff2 / ws.Cells(i - 1, curr2Col).Value
            End If
        End If
    Next i
End Sub

Sub CalculateWeekly(ws As Worksheet, startRow As Long, lastRow As Long, dateCol As Long, curr1Col As Long, curr2Col As Long, outputCol As Long)
    Dim i As Long
    Dim weekStart As Long
    
    i = startRow
    Do While i <= lastRow
        weekStart = i
        ' Find the end of the week (next 4 rows or end of data)
        Dim weekEnd As Long
        weekEnd = Application.Min(i + 4, lastRow) ' 5 days total (i to i+4)
        
        ' Skip if not enough data for a full week
        If weekEnd > weekStart And ws.Cells(weekStart, curr1Col).Value <> "" And ws.Cells(weekEnd, curr1Col).Value <> "" Then
            ' Calculate absolute differences for the week
            Dim absDiff1 As Double, absDiff2 As Double
            absDiff1 = Abs(ws.Cells(weekEnd, curr1Col).Value - ws.Cells(weekStart, curr1Col).Value)
            absDiff2 = Abs(ws.Cells(weekEnd, curr2Col).Value - ws.Cells(weekStart, curr2Col).Value)
            
            ' Fill the values for all days in this week
            Dim j As Long
            For j = weekStart To weekEnd
                ws.Cells(j, outputCol).Value = absDiff1
                ws.Cells(j, outputCol + 1).Value = absDiff2
                
                ' Calculate relative changes
                If ws.Cells(weekStart, curr1Col).Value <> 0 Then
                    ws.Cells(j, outputCol + 2).Value = absDiff1 / ws.Cells(weekStart, curr1Col).Value
                End If
                If ws.Cells(weekStart, curr2Col).Value <> 0 Then
                    ws.Cells(j, outputCol + 3).Value = absDiff2 / ws.Cells(weekStart, curr2Col).Value
                End If
            Next j
        End If
        
        ' Move to next week
        i = weekEnd + 1
    Loop
End Sub

Sub CalculateMonthly(ws As Worksheet, startRow As Long, lastRow As Long, dateCol As Long, curr1Col As Long, curr2Col As Long, outputCol As Long)
    Dim i As Long
    Dim currentMonth As Long, currentYear As Long
    Dim monthStart As Long, monthEnd As Long
    
    i = startRow
    Do While i <= lastRow
        If ws.Cells(i, dateCol).Value <> "" Then
            currentMonth = Month(ws.Cells(i, dateCol).Value)
            currentYear = Year(ws.Cells(i, dateCol).Value)
            monthStart = i
            
            ' Find the end of the month
            Do While i <= lastRow And ws.Cells(i, dateCol).Value <> ""
                If Month(ws.Cells(i, dateCol).Value) <> currentMonth Or Year(ws.Cells(i, dateCol).Value) <> currentYear Then
                    Exit Do
                End If
                i = i + 1
            Loop
            monthEnd = i - 1
            
            ' Calculate monthly differences
            If monthEnd > monthStart And ws.Cells(monthStart, curr1Col).Value <> "" And ws.Cells(monthEnd, curr1Col).Value <> "" Then
                Dim absDiff1 As Double, absDiff2 As Double
                absDiff1 = Abs(ws.Cells(monthEnd, curr1Col).Value - ws.Cells(monthStart, curr1Col).Value)
                absDiff2 = Abs(ws.Cells(monthEnd, curr2Col).Value - ws.Cells(monthStart, curr2Col).Value)
                
                ' Fill values for all days in this month
                Dim j As Long
                For j = monthStart To monthEnd
                    ws.Cells(j, outputCol).Value = absDiff1
                    ws.Cells(j, outputCol + 1).Value = absDiff2
                    
                    ' Calculate relative changes
                    If ws.Cells(monthStart, curr1Col).Value <> 0 Then
                        ws.Cells(j, outputCol + 2).Value = absDiff1 / ws.Cells(monthStart, curr1Col).Value
                    End If
                    If ws.Cells(monthStart, curr2Col).Value <> 0 Then
                        ws.Cells(j, outputCol + 3).Value = absDiff2 / ws.Cells(monthStart, curr2Col).Value
                    End If
                Next j
            End If
        Else
            i = i + 1
        End If
    Loop
End Sub

Sub CalculateQuarterly(ws As Worksheet, startRow As Long, lastRow As Long, dateCol As Long, curr1Col As Long, curr2Col As Long, outputCol As Long)
    Dim i As Long
    Dim currentQuarter As Long, currentYear As Long
    Dim quarterStart As Long, quarterEnd As Long
    
    i = startRow
    Do While i <= lastRow
        If ws.Cells(i, dateCol).Value <> "" Then
            currentQuarter = Int((Month(ws.Cells(i, dateCol).Value) - 1) / 3) + 1
            currentYear = Year(ws.Cells(i, dateCol).Value)
            quarterStart = i
            
            ' Find the end of the quarter
            Do While i <= lastRow And ws.Cells(i, dateCol).Value <> ""
                Dim thisQuarter As Long
                thisQuarter = Int((Month(ws.Cells(i, dateCol).Value) - 1) / 3) + 1
                If thisQuarter <> currentQuarter Or Year(ws.Cells(i, dateCol).Value) <> currentYear Then
                    Exit Do
                End If
                i = i + 1
            Loop
            quarterEnd = i - 1
            
            ' Calculate quarterly differences
            If quarterEnd > quarterStart And ws.Cells(quarterStart, curr1Col).Value <> "" And ws.Cells(quarterEnd, curr1Col).Value <> "" Then
                Dim absDiff1 As Double, absDiff2 As Double
                absDiff1 = Abs(ws.Cells(quarterEnd, curr1Col).Value - ws.Cells(quarterStart, curr1Col).Value)
                absDiff2 = Abs(ws.Cells(quarterEnd, curr2Col).Value - ws.Cells(quarterStart, curr2Col).Value)
                
                ' Fill values for all days in this quarter
                Dim j As Long
                For j = quarterStart To quarterEnd
                    ws.Cells(j, outputCol).Value = absDiff1
                    ws.Cells(j, outputCol + 1).Value = absDiff2
                    
                    ' Calculate relative changes
                    If ws.Cells(quarterStart, curr1Col).Value <> 0 Then
                        ws.Cells(j, outputCol + 2).Value = absDiff1 / ws.Cells(quarterStart, curr1Col).Value
                    End If
                    If ws.Cells(quarterStart, curr2Col).Value <> 0 Then
                        ws.Cells(j, outputCol + 3).Value = absDiff2 / ws.Cells(quarterStart, curr2Col).Value
                    End If
                Next j
            End If
        Else
            i = i + 1
        End If
    Loop
End Sub
