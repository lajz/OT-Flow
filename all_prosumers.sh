#!/bin/bash

declare -a StringArrayFull=(
	"Hog_services_Kerrie.csv"
	"Hog_assembly_Colette.csv"
	"Hog_assembly_Dona.csv"
	"Hog_assembly_Jasmine.csv"
	"Hog_education_Casandra.csv"
	"Hog_education_Donnie.csv"
	"Hog_education_Jewel.csv"
	"Hog_education_Jordan.csv"
	"Hog_education_Madge.csv"
	"Hog_education_Rachael.csv"
	"Hog_food_Morgan.csv"
	"Hog_industrial_Jeremy.csv"
	"Hog_industrial_Joanne.csv"
	"Hog_industrial_Mariah.csv"
	"Hog_industrial_Quentin.csv"
	"Hog_lodging_Francisco.csv"
	"Hog_lodging_Hal.csv"
	"Hog_lodging_Nikki.csv"
	"Hog_lodging_Ora.csv"
	"Hog_lodging_Shanti.csv"
	"Hog_office_Almeda.csv"
	"Hog_office_Bessie.csv"
	"Hog_office_Betsy.csv"
	"Hog_office_Bill.csv"
	"Hog_office_Denita.csv"
	"Hog_office_Gustavo.csv"
	"Hog_office_Lavon.csv"
	"Hog_office_Lizzie.csv"
	"Hog_office_Mary.csv"
	"Hog_office_Merilyn.csv"
	"Hog_office_Mike.csv"
	"Hog_office_Miriam.csv"
	"Hog_office_Myles.csv"
	"Hog_office_Napoleon.csv"
	"Hog_office_Shawna.csv"
	"Hog_office_Shawnna.csv"
	"Hog_office_Sherrie.csv"
	"Hog_office_Shon.csv"
	"Hog_office_Sydney.csv"
	"Hog_office_Valda.csv"
	"Hog_other_Noma.csv"
	"Hog_other_Tobias.csv"
	"Hog_parking_Jean.csv"
	"Hog_parking_Shannon.csv"
	"Hog_public_Crystal.csv"
	"Hog_public_Gerard.csv"
	"Hog_public_Kevin.csv"
	"Hog_public_Octavia.csv"
	"Hog_services_Adrianna.csv" 
)

declare -a StringArray=(
	"Hog_assembly_Jasmine.csv"
	"Hog_education_Casandra.csv"
)

for val in ${StringArrayFull[@]}; do
	echo $i
	sbatch singular_prosumer.sh $val $1

done
