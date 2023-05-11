*FILE '$SR3FILE'  ** Nome do arquivo de entrada
*SPREADSHEET
*TIME ON                 ** The tables will have no time column,
*TABLE-FOR

   *COLUMN-FOR  *GROUPS 'Plataforma-PRO'
				*PARAMETERS 'Period Oil Production - Monthly SC'
   *COLUMN-FOR  *GROUPS 'Plataforma-PRO'
				*PARAMETERS 'Period Gas Production - Monthly SC'
   *COLUMN-FOR  *GROUPS 'Plataforma-PRO'
				*PARAMETERS 'Period Water Production - Monthly SC'
   *COLUMN-FOR  *GROUPS 'Plataforma-INJ'
				*PARAMETERS 'Period Water Production - Monthly SC'
   *COLUMN-FOR  *GROUPS 'Plataforma-PRO'
				*PARAMETERS 'Liquid Rate SC'

*TABLE-END

** *LIST-PARAMETERS causes Results Report to list all the allowed parameters and origins for columns in a table (for first opened file).
