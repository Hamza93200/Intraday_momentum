Sub CalculateCurrencyReturns()
    Dim wsSpot As Worksheet
    Dim wsReturns As Worksheet
    Dim lastRow As Long
    Dim i As Long, j As Long
    Dim currentCol As Long
    
    ' Set worksheets
    Set wsSpot = ThisWorkbook.Sheets("Spot")
    Set wsReturns = ThisWorkbook.Sheets("Returns")
    
    ' Clear existing data in Returns sheet
    wsReturns.Cells.Clear
    
    ' Get last row of data in Spot sheet
    lastRow = wsSpot.Cells(wsSpot.Rows.Count, 1).End(xlUp).Row
    
    ' Start column counter for Returns sheet
    currentCol = 1
    
    ' Get currency names (assuming they are in row 1)
    Dim currency1 As String, currency2 As String
    currency1 = wsSpot.Cells(1, 2).Value ' Second column header
    currency2 = wsSpot.Cells(1, 3).Value ' Third column header
    
    ' DAILY CALCULATIONS
    ' Headers for daily
    wsReturns.Cells(1, currentCol).Value = "Date"
    wsReturns.Cells(1, currentCol + 1).Value = currency1 & " Daily Abs Diff"
    wsReturns.Cells(1, currentCol + 2).Value = currency2 & " Daily Abs Diff"
    wsReturns.Cells(1, currentCol + 3).Value = currency1 & " Daily Return %"
    wsReturns.Cells(1, currentCol + 4).Value = currency2 & " Daily Return %"
    
    ' Calculate daily returns
    For i = 2 To lastRow
        If i > 2 Then ' Start from row 3 since we need previous day
            wsReturns.Cells(i - 1, currentCol).Value = wsSpot.Cells(i, 1).Value ' Date
            
            ' Absolute differences
            wsReturns.Cells(i - 1, currentCol + 1).Value = Abs(wsSpot.Cells(i, 2).Value - wsSpot.Cells(i - 1, 2).Value)
            wsReturns.Cells(i - 1, currentCol + 2).Value = Abs(wsSpot.Cells(i, 3).Value - wsSpot.Cells(i - 1, 3).Value)
            
            ' Percentage returns
            If wsSpot.Cells(i - 1, 2).Value <> 0 Then
                wsReturns.Cells(i - 1, currentCol + 3).Value = Abs(wsSpot.Cells(i, 2).Value - wsSpot.Cells(i - 1, 2).Value) / wsSpot.Cells(i - 1, 2).Value
            End If
            If wsSpot.Cells(i - 1, 3).Value <> 0 Then
                wsReturns.Cells(i - 1, currentCol + 4).Value = Abs(wsSpot.Cells(i, 3).Value - wsSpot.Cells(i - 1, 3).Value) / wsSpot.Cells(i - 1, 3).Value
            End If
        End If
    Next i
    
    currentCol = currentCol + 6 ' Move to next set of columns
    
    ' WEEKLY CALCULATIONS
    ' Headers for weekly
    wsReturns.Cells(1, currentCol).Value = "Week"
    wsReturns.Cells(1, currentCol + 1).Value = currency1 & " Weekly Abs Diff"
    wsReturns.Cells(1, currentCol + 2).Value = currency2 & " Weekly Abs Diff"
    wsReturns.Cells(1, currentCol + 3).Value = currency1 & " Weekly Return %"
    wsReturns.Cells(1, currentCol + 4).Value = currency2 & " Weekly Return %"
    
    ' Calculate weekly returns (5 business days)
    Dim weekCounter As Long
    weekCounter = 1
    Dim resultsRow As Long
    resultsRow = 2
    
    For i = 2 To lastRow - 4 Step 5 ' Step by 5 days for weekly
        If i + 4 <= lastRow Then
            wsReturns.Cells(resultsRow, currentCol).Value = "Week " & weekCounter
            
            ' Absolute differences (T+5 - T)
            wsReturns.Cells(resultsRow, currentCol + 1).Value = Abs(wsSpot.Cells(i + 4, 2).Value - wsSpot.Cells(i, 2).Value)
            wsReturns.Cells(resultsRow, currentCol + 2).Value = Abs(wsSpot.Cells(i + 4, 3).Value - wsSpot.Cells(i, 3).Value)
            
            ' Percentage returns
            If wsSpot.Cells(i, 2).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 3).Value = Abs(wsSpot.Cells(i + 4, 2).Value - wsSpot.Cells(i, 2).Value) / wsSpot.Cells(i, 2).Value
            End If
            If wsSpot.Cells(i, 3).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 4).Value = Abs(wsSpot.Cells(i + 4, 3).Value - wsSpot.Cells(i, 3).Value) / wsSpot.Cells(i, 3).Value
            End If
            
            weekCounter = weekCounter + 1
            resultsRow = resultsRow + 1
        End If
    Next i
    
    currentCol = currentCol + 6 ' Move to next set of columns
    
    ' MONTHLY CALCULATIONS (assuming 22 business days per month)
    ' Headers for monthly
    wsReturns.Cells(1, currentCol).Value = "Month"
    wsReturns.Cells(1, currentCol + 1).Value = currency1 & " Monthly Abs Diff"
    wsReturns.Cells(1, currentCol + 2).Value = currency2 & " Monthly Abs Diff"
    wsReturns.Cells(1, currentCol + 3).Value = currency1 & " Monthly Return %"
    wsReturns.Cells(1, currentCol + 4).Value = currency2 & " Monthly Return %"
    
    ' Calculate monthly returns
    Dim monthCounter As Long
    monthCounter = 1
    resultsRow = 2
    
    For i = 2 To lastRow - 21 Step 22 ' Step by 22 days for monthly
        If i + 21 <= lastRow Then
            wsReturns.Cells(resultsRow, currentCol).Value = "Month " & monthCounter
            
            ' Absolute differences
            wsReturns.Cells(resultsRow, currentCol + 1).Value = Abs(wsSpot.Cells(i + 21, 2).Value - wsSpot.Cells(i, 2).Value)
            wsReturns.Cells(resultsRow, currentCol + 2).Value = Abs(wsSpot.Cells(i + 21, 3).Value - wsSpot.Cells(i, 3).Value)
            
            ' Percentage returns
            If wsSpot.Cells(i, 2).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 3).Value = Abs(wsSpot.Cells(i + 21, 2).Value - wsSpot.Cells(i, 2).Value) / wsSpot.Cells(i, 2).Value
            End If
            If wsSpot.Cells(i, 3).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 4).Value = Abs(wsSpot.Cells(i + 21, 3).Value - wsSpot.Cells(i, 3).Value) / wsSpot.Cells(i, 3).Value
            End If
            
            monthCounter = monthCounter + 1
            resultsRow = resultsRow + 1
        End If
    Next i
    
    currentCol = currentCol + 6 ' Move to next set of columns
    
    ' QUARTERLY CALCULATIONS (assuming 66 business days per quarter)
    ' Headers for quarterly
    wsReturns.Cells(1, currentCol).Value = "Quarter"
    wsReturns.Cells(1, currentCol + 1).Value = currency1 & " Quarterly Abs Diff"
    wsReturns.Cells(1, currentCol + 2).Value = currency2 & " Quarterly Abs Diff"
    wsReturns.Cells(1, currentCol + 3).Value = currency1 & " Quarterly Return %"
    wsReturns.Cells(1, currentCol + 4).Value = currency2 & " Quarterly Return %"
    
    ' Calculate quarterly returns
    Dim quarterCounter As Long
    quarterCounter = 1
    resultsRow = 2
    
    For i = 2 To lastRow - 65 Step 66 ' Step by 66 days for quarterly
        If i + 65 <= lastRow Then
            wsReturns.Cells(resultsRow, currentCol).Value = "Quarter " & quarterCounter
            
            ' Absolute differences
            wsReturns.Cells(resultsRow, currentCol + 1).Value = Abs(wsSpot.Cells(i + 65, 2).Value - wsSpot.Cells(i, 2).Value)
            wsReturns.Cells(resultsRow, currentCol + 2).Value = Abs(wsSpot.Cells(i + 65, 3).Value - wsSpot.Cells(i, 3).Value)
            
            ' Percentage returns
            If wsSpot.Cells(i, 2).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 3).Value = Abs(wsSpot.Cells(i + 65, 2).Value - wsSpot.Cells(i, 2).Value) / wsSpot.Cells(i, 2).Value
            End If
            If wsSpot.Cells(i, 3).Value <> 0 Then
                wsReturns.Cells(resultsRow, currentCol + 4).Value = Abs(wsSpot.Cells(i + 65, 3).Value - wsSpot.Cells(i, 3).Value) / wsSpot.Cells(i, 3).Value
            End If
            
            quarterCounter = quarterCounter + 1
            resultsRow = resultsRow + 1
        End If
    Next i
    
    ' Format the Returns sheet
    With wsReturns.Range("A1").CurrentRegion
        .HorizontalAlignment = xlCenter
        .Font.Bold = True
    End With
    
    ' Format percentage columns
    For i = 4 To currentCol + 4 Step 5 ' Every 5th column starting from column 4
        If i <= wsReturns.Cells(1, wsReturns.Columns.Count).End(xlToLeft).Column Then
            wsReturns.Columns(i).NumberFormat = "0.00%"
            If i + 1 <= wsReturns.Cells(1, wsReturns.Columns.Count).End(xlToLeft).Column Then
                wsReturns.Columns(i + 1).NumberFormat = "0.00%"
            End If
        End If
    Next i
    
    ' Auto-fit columns
    wsReturns.Cells.EntireColumn.AutoFit
    
    MsgBox "Currency returns calculations completed successfully!", vbInformation, "Process Complete"
    
End Sub
