import {Container, Grid} from '@mui/material'


function Result(){
    return(
        <Container maxWidth="md" style={{height:"100vh"}}>
            <Grid container direction="column" justifyContent="center" alignItems="center">
                <Grid item style={{marginTop:"24px"}}>文字結果</Grid>
                <Grid item container direction="row" justifyContent="space-between" alignItems="center" style={{marginTop:"24px"}}>
                    <Grid item>圖1</Grid>
                    <Grid item>圖2</Grid>
                </Grid>
                <Grid item container direction="row" justifyContent="space-between" alignItems="center" style={{marginTop:"24px"}}>
                    <Grid item>圖3</Grid>
                    <Grid item>圖4</Grid>
                </Grid>
            </Grid>
        </Container>
    )
}

export default Result;